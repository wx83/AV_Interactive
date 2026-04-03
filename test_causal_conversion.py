#!/usr/bin/env python3
"""Test script validating the 5-step conversion from bidirectional LTX-2
to causal Infinity-RoPE-enabled LTX-2, plus a self-forcing + DMD training
step with dummy data.

Steps:
  1. Convert bidirectional attention → causal attention
  2. Load pretrained weights into causal model
  3. Replace absolute RoPE with window-relative RoPE
  4. Convert global timestep to per-frame timestep
  5. Wire up Self-Forcing + DMD components
  6. Run one self-forcing + DMD training step (forward + backward)
"""

import sys
import os
import traceback

# Add LTX-2 packages to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-core", "src"))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-trainer", "src"))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-pipelines", "src"))

import torch

# ── shared small config ──────────────────────────────────────────────
SMALL_MODEL_KWARGS = dict(
    num_attention_heads=4,
    attention_head_dim=32,   # inner_dim = 4*32 = 128
    in_channels=32,
    out_channels=32,
    num_layers=2,
    cross_attention_dim=128, # must match inner_dim when no caption_projection
    positional_embedding_max_pos=[8, 16, 16],
    timestep_scale_multiplier=1000,
)


# =====================================================================
# Step 1: Convert bidirectional attention → causal attention
# =====================================================================
def step1_convert_attention():
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.causal_model import CausalLTXModel
    from ltx_core.model.transformer.ltx_wrapper import LTXDiffusionWrapper
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
    from ltx_core.model.transformer.causal_transformer import CausalAVTransformerBlock
    from ltx_core.model.transformer.attention import Attention
    from ltx_core.model.transformer.causal_attention import CausalAttention

    # Bidirectional
    bidir = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )
    # Causal
    causal = LTXDiffusionWrapper(
        is_causal=True,
        num_frame_per_block=2,
        local_attn_size=4, # four frames that are used for attention
        sink_size=1, # the first frame is always kept
        model_kwargs=dict(model_type="ltx video only model", **SMALL_MODEL_KWARGS),
    )

    assert isinstance(bidir.model, LTXModel), f"Expected LTXModel, got {type(bidir.model)}"
    assert isinstance(causal.model, CausalLTXModel), f"Expected CausalLTXModel, got {type(causal.model)}"

    # Check transformer block types
    b_block = bidir.model.transformer_blocks[0]
    c_block = causal.model.transformer_blocks[0]
    assert isinstance(b_block, BasicAVTransformerBlock), f"Expected BasicAVTransformerBlock, got {type(b_block)}"
    assert isinstance(c_block, CausalAVTransformerBlock), f"Expected CausalAVTransformerBlock, got {type(c_block)}"

    # Check attention types
    assert isinstance(b_block.attn1, Attention), f"Expected Attention, got {type(b_block.attn1)}"
    assert isinstance(c_block.attn1, CausalAttention), f"Expected CausalAttention, got {type(c_block.attn1)}"

    # Verify causal-specific attributes
    assert hasattr(causal.model, "local_attn_size")
    assert hasattr(causal.model, "sink_size")
    assert hasattr(causal.model, "num_frame_per_block")

    print("  Bidirectional: LTXModel + BasicAVTransformerBlock + Attention")
    print("  Causal:        CausalLTXModel + CausalAVTransformerBlock + CausalAttention")
    return bidir, causal


# =====================================================================
# Step 2: Load pretrained weights into causal model
# =====================================================================
def step2_load_weights(bidir, causal):
    # Initialise bidirectional weights (random, since on meta device we
    # need to materialise first)
    bidir_sd = bidir.model.state_dict()
    result = causal.model.load_state_dict(bidir_sd, strict=False)

    if result.unexpected_keys:
        print(f"  WARNING: unexpected keys: {result.unexpected_keys}")
    assert len(result.unexpected_keys) == 0, (
        f"Unexpected keys in causal model: {result.unexpected_keys}"
    )

    if result.missing_keys:
        print(f"  Missing keys (causal-only params): {result.missing_keys}")
    else:
        print("  All weights transferred with zero missing/unexpected keys")

    # Spot-check a few weights
    bidir_sd2 = bidir.model.state_dict()
    causal_sd = causal.model.state_dict()
    for key in ["patchify_proj.weight", "transformer_blocks.0.attn2.to_q.weight"]:
        if key in bidir_sd2 and key in causal_sd:
            assert torch.equal(bidir_sd2[key], causal_sd[key]), f"Mismatch on {key}"
            print(f"  Verified: {key}")


# =====================================================================
# Step 3: Replace absolute RoPE with window-relative RoPE
# =====================================================================
def step3_test_rope():
    from ltx_core.model.transformer.causal_rope import (
        block_relative_positions,
        build_window_positions,
        rope_cut_positions,
    )

    # 3a: build_window_positions
    pos = build_window_positions(
        num_window_frames=4, frame_height=2, frame_width=2,
        batch_size=1, with_bounds=True,
    )
    assert pos.shape == (1, 3, 16, 2), f"Expected (1,3,16,2), got {pos.shape}"
    temporal_starts = pos[0, 0, :, 0]  # start bounds of temporal dim
    expected_t = torch.tensor([0]*4 + [1]*4 + [2]*4 + [3]*4)
    assert torch.equal(temporal_starts, expected_t), (
        f"Temporal starts mismatch: {temporal_starts}"
    )
    print(f"  build_window_positions: shape={pos.shape}, temporal OK")

    # 3b: block_relative_positions
    rel = block_relative_positions(
        positions=pos,
        start_frame=2,
        tokens_per_frame=4,
        max_temporal_pos=8,
    )
    rel_t = rel[0, 0, :, 0]
    # Original: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
    # After shift by -2 and clamp [0,7]: [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1]
    expected_rel = torch.tensor([0]*4 + [0]*4 + [0]*4 + [1]*4)
    assert torch.equal(rel_t, expected_rel), (
        f"Relative positions mismatch: {rel_t} vs {expected_rel}"
    )
    print(f"  block_relative_positions: shift by start_frame=2, clamped OK")

    # 3c: rope_cut_positions
    cut = rope_cut_positions(
        positions=pos,
        tokens_per_frame=4,
        transition_frames=1,
        transition_from=45,
    )
    cut_t = cut[0, 0, -4:, 0]  # last frame temporal starts
    assert (cut_t == 45).all(), f"Expected 45 for last frame, got {cut_t}"
    print(f"  rope_cut_positions: last frame temporal → 45 OK")


# =====================================================================
# Step 4: Convert global timestep to per-frame timestep
# =====================================================================
def step4_test_timesteps():
    from ltx_core.model.transformer.model import LTXModelType
    from ltx_core.model.transformer.ltx_wrapper import LTXDiffusionWrapper

    bidir = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )
    causal = LTXDiffusionWrapper(
        is_causal=True,
        num_frame_per_block=2,
        model_kwargs=dict(model_type="ltx video only model", **SMALL_MODEL_KWARGS),
    )

    torch.manual_seed(42)

    # Bidirectional: uniform timestep (same across all frames)
    t_bidir = bidir.get_timestep(batch_size=2, num_frames=8, num_frame_per_block=2)
    assert t_bidir.shape == (2, 8), f"Expected (2,8), got {t_bidir.shape}"
    # All frames in a batch element should be the same
    assert (t_bidir[0] == t_bidir[0, 0]).all(), "Bidirectional timesteps should be uniform"
    print(f"  Bidirectional: shape={t_bidir.shape}, all frames identical per sample")

    # Causal: per-frame timestep (varies across blocks, uniform within block)
    t_causal = causal.get_timestep(batch_size=2, num_frames=8, num_frame_per_block=2)
    assert t_causal.shape == (2, 8), f"Expected (2,8), got {t_causal.shape}"
    # Within each block of 2, values should match
    for b in range(2):
        for block_start in range(0, 8, 2):
            assert t_causal[b, block_start] == t_causal[b, block_start + 1], (
                f"Within-block mismatch at batch={b}, block_start={block_start}"
            )
    print(f"  Causal: shape={t_causal.shape}, per-block varying, within-block uniform")

    # Verify they differ (with high probability)
    if not (t_causal[0, 0] == t_causal[0, 2]):
        print(f"  Confirmed: different blocks have different timesteps")
    else:
        print(f"  Note: blocks happened to get same timestep (rare but possible)")


# =====================================================================
# Step 5: Wire up Self-Forcing + DMD
# =====================================================================
def step5_test_self_forcing_dmd():
    from ltx_core.model.transformer.model import LTXModelType
    from ltx_core.model.transformer.ltx_wrapper import LTXDiffusionWrapper
    from ltx_trainer.dmd import DMDLoss
    from ltx_trainer.self_forcing_pipeline import SelfForcingTrainingPipeline

    gen = LTXDiffusionWrapper(
        is_causal=True,
        num_frame_per_block=2,
        local_attn_size=4,
        sink_size=1,
        model_kwargs=dict(model_type="ltx video only model", **SMALL_MODEL_KWARGS),
    )
    real_score = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )
    fake_score = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )

    # Freeze real_score (as in real training)
    for p in real_score.parameters(): # real score is not trainable but fake score is trainable
        p.requires_grad_(False)
    # fake score learns to track the generator's evloving output distribution
    # DMD loss
    dmd = DMDLoss(
        generator=gen,
        real_score=real_score,
        fake_score=fake_score,
        num_train_timestep=1000,
        real_guidance_scale=4.5,
        fake_guidance_scale=0.0,
        num_frame_per_block=2,
    )
    assert dmd.generator is gen
    assert dmd.real_score is real_score
    assert dmd.fake_score is fake_score
    print("  DMDLoss: wired generator + real_score + fake_score")

    # Self-Forcing pipeline
    tokens_per_frame = 4  # 2x2 spatial grid
    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=[1.0, 0.75, 0.5, 0.25],
        generator=gen,
        num_frame_per_block=2,
        num_max_frames=8,
        tokens_per_frame=tokens_per_frame,
    )
    assert pipeline.generator is gen
    assert pipeline.num_frame_per_block == 2
    assert pipeline.num_layers == 2
    assert pipeline.tokens_per_frame == tokens_per_frame
    print(f"  SelfForcingTrainingPipeline: num_layers={pipeline.num_layers}, "
          f"num_frame_per_block={pipeline.num_frame_per_block}")

    # Verify trainable param counts
    gen_trainable = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    real_trainable = sum(p.numel() for p in real_score.parameters() if p.requires_grad)
    fake_trainable = sum(p.numel() for p in fake_score.parameters() if p.requires_grad)
    print(f"  Generator params (trainable): {gen_trainable:,}")
    print(f"  Real score params (frozen):   {real_trainable:,} (should be 0)")
    print(f"  Fake score params (trainable): {fake_trainable:,}")
    assert real_trainable == 0, "Real score should be fully frozen"


# =====================================================================
# Step 6: Run one self-forcing + DMD training step
# =====================================================================
def step6_training_step():
    """Mimics infinity-rope's distillation training loop with dummy data.

    Like infinity-rope, the only real input is text embeddings (from prompts).
    Video is generated entirely from noise via backward simulation.

    Uses the causal model's _forward_train path (block-causal masking, no
    KV cache) for the generator, and bidirectional models for the scores.
    This matches the actual training setup where self-forcing generates
    video and DMD computes the distillation loss.
    """
    from ltx_core.model.transformer.model import LTXModelType
    from ltx_core.model.transformer.ltx_wrapper import LTXDiffusionWrapper
    from ltx_core.model.transformer.modality import Modality
    from ltx_trainer.dmd import DMDLoss

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  WARNING: CUDA not available. flex_attention backward requires GPU.")
        print("  Run with: srun -p sharedp --gres=gpu:1 bash -c 'conda run -n ltx2 python test_causal_conversion.py'")
        raise RuntimeError("Step 6 requires CUDA for flex_attention backward pass")

    # ── Config ──
    B = 1                    # batch size
    num_frames = 4           # total frames to generate
    num_frame_per_block = 2  # frames per causal block
    in_channels = 32         # latent channels (matches SMALL_MODEL_KWARGS)
    inner_dim = 128          # 4 heads * 32 dim
    cross_attention_dim = 128  # must match inner_dim (no caption_projection)
    H, W = 2, 2             # spatial grid per frame
    tokens_per_frame = H * W  # 4
    total_tokens = num_frames * tokens_per_frame  # 16

    # ── Create 3 models (like infinity-rope's BaseModel._initialize_models) ──
    gen = LTXDiffusionWrapper(
        is_causal=True,
        num_frame_per_block=num_frame_per_block,
        local_attn_size=4,
        sink_size=1,
        model_kwargs=dict(model_type="ltx video only model", **SMALL_MODEL_KWARGS),
    )
    real_score = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )
    fake_score = LTXDiffusionWrapper(
        is_causal=False,
        model_kwargs=dict(model_type=LTXModelType.VideoOnly, **SMALL_MODEL_KWARGS),
    )

    # Move to GPU
    gen = gen.to(device)
    real_score = real_score.to(device)
    fake_score = fake_score.to(device)

    # Freeze real_score (pretrained teacher)
    for p in real_score.parameters():
        p.requires_grad_(False)

    # ── DMD loss ──
    dmd = DMDLoss(
        generator=gen,
        real_score=real_score,
        fake_score=fake_score,
        num_train_timestep=1000,
        real_guidance_scale=4.5,
        fake_guidance_scale=0.0,
        num_frame_per_block=num_frame_per_block,
    )

    # ── Optimizers ──
    gen_optim = torch.optim.AdamW(
        [p for p in gen.parameters() if p.requires_grad], lr=1e-4,
    )
    critic_optim = torch.optim.AdamW(
        [p for p in fake_score.parameters() if p.requires_grad], lr=1e-4,
    )

    # ── Dummy data ──
    # In real training: text_encoder(["a cat running..."]) → video_context
    ctx_len = 16
    video_context = torch.randn(B, ctx_len, cross_attention_dim, device=device)
    video_context_neg = torch.zeros(B, ctx_len, cross_attention_dim, device=device)
    positions = torch.zeros(B, 3, total_tokens, 2, device=device)

    # ── Generator step: simulate self-forcing output ──
    # In real training, the self-forcing pipeline generates video block-by-block
    # via the KV-cache inference path. Here we use the _forward_train path
    # (block-causal masking) which is equivalent but avoids in-place cache ops.
    print("  Running generator forward (causal, _forward_train path)...")
    gen_optim.zero_grad(set_to_none=True)

    # Create noisy input (simulating self-forcing output)
    noise_latent = torch.randn(B, total_tokens, in_channels, device=device)
    sigma_val = 0.5
    timesteps = torch.full((B, total_tokens), sigma_val, device=device)

    gen_mod = Modality(
        latent=noise_latent,
        sigma=torch.tensor([sigma_val], device=device).expand(B),
        timesteps=timesteps,
        positions=positions,
        context=video_context,
        enabled=True,
    )

    # Causal generator forward (uses _forward_train, no KV cache)
    vx, _ = gen(video=gen_mod, audio=None)
    from ltx_core.utils import to_denoised
    generated_video = to_denoised(noise_latent, vx, sigma_val)
    print(f"  Generator output: {generated_video.shape}")  # (B, total_tokens, in_channels)

    # DMD generator loss
    gen_loss, gen_log = dmd.generator_loss(
        generated_video=generated_video,
        video_positions=positions,
        video_context_pos=video_context,
        video_context_neg=video_context_neg,
    )
    print(f"  Generator loss: {gen_loss.item():.6f}")

    gen_loss.backward()
    gen_grad_norm = torch.nn.utils.clip_grad_norm_(gen.parameters(), 10.0)
    gen_optim.step()
    print(f"  Generator grad norm: {gen_grad_norm:.6f}")

    # ── Critic step ──
    print("  Running critic step...")
    critic_optim.zero_grad(set_to_none=True)

    with torch.no_grad():
        noise2 = torch.randn(B, total_tokens, in_channels, device=device)
        gen_mod2 = Modality(
            latent=noise2,
            sigma=torch.tensor([sigma_val], device=device).expand(B),
            timesteps=torch.full((B, total_tokens), sigma_val, device=device),
            positions=positions,
            context=video_context,
            enabled=True,
        )
        vx2, _ = gen(video=gen_mod2, audio=None)
        generated_video2 = to_denoised(noise2, vx2, sigma_val)

    critic_loss, critic_log = dmd.critic_loss(
        generated_video=generated_video2,
        video_positions=positions,
        video_context_pos=video_context,
    )
    print(f"  Critic loss: {critic_loss.item():.6f}")

    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(fake_score.parameters(), 10.0)
    critic_optim.step()
    print(f"  Critic grad norm: {critic_grad_norm:.6f}")

    print(f"  Training step completed successfully (generator + critic)")


# =====================================================================
# Main
# =====================================================================
def main():
    steps = [
        ("Step 1: Bidirectional → Causal attention", step1_convert_attention),
        ("Step 2: Load pretrained weights", None),  # needs args
        ("Step 3: Absolute RoPE → Window-relative RoPE", step3_test_rope),
        ("Step 4: Global timestep → Per-frame timestep", step4_test_timesteps),
        ("Step 5: Self-Forcing + DMD wiring", step5_test_self_forcing_dmd),
        ("Step 6: Self-Forcing + DMD training step", step6_training_step),
    ]

    results = {}
    bidir_model = None
    causal_model = None

    for name, fn in steps:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            if name.startswith("Step 1"):
                bidir_model, causal_model = step1_convert_attention()
                results[name] = "PASS"
            elif name.startswith("Step 2"):
                assert bidir_model is not None, "Step 1 must pass first"
                step2_load_weights(bidir_model, causal_model)
                results[name] = "PASS"
            else:
                fn()
                results[name] = "PASS"
            print(f"  >> PASS")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  >> FAIL: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result == "PASS" else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {name}")

    if all_pass:
        print(f"\nAll 6 steps passed!")
    else:
        print(f"\nSome steps failed.")
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
