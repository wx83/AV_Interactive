#!/usr/bin/env python3
"""Real-time action-conditioned streaming AV generation demo.

Uses the CausalLTXModel with KV caches so each block's generation is
temporally conditioned on all previous blocks.  Action prompts can be
injected at specified block indices to steer the continuation.

The generation flow:
  1. Encode all prompts (initial + negative + action) via Gemma.
  2. Load pretrained LTXModel weights → create CausalLTXModel (same weights).
  3. For each block:
     a. Run multi-step Euler denoising on this block's latent tokens.
        Between steps, the KV cache is rewound so only the *final* step's
        K/V populate the cache for future blocks.
     b. Action context can switch at any block boundary.
  4. Accumulate all denoised latents, decode with video/audio VAE.
  5. Save the combined .mp4.

Usage:
    python test_realtime_action_stream.py \\
        --checkpoint-path models/ltx-2.3-22b-dev.safetensors \\
        --gemma-root models/gemma-3-12b-it-qat-q4_0-unquantized \\
        --initial-prompt "A person walking in a forest" \\
        --action-prompts "The person stops and looks up" "A bird flies overhead" \\
        --action-at-block 5 10 \\
        --num-blocks 15 \\
        --output-path output_stream.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-core", "src"))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-pipelines", "src"))

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.transformer.causal_model import CausalLTXModel
from ltx_core.model.transformer.causal_rope import build_window_positions
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.model.video_vae.tiling import TilingConfig, TemporalTilingConfig
from ltx_core.types import (
    Audio,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_core.utils import to_denoised
from ltx_pipelines.utils import ModelLedger, cleanup_memory, encode_prompts
from ltx_pipelines.utils.constants import VIDEO_LATENT_CHANNELS, VIDEO_SCALE_FACTORS
from ltx_pipelines.utils.media_io import encode_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position builders (from test_causal_stress.py, adapted)
# ---------------------------------------------------------------------------

def build_video_block_positions(
    batch_size: int, start_frame: int, num_frames: int,
    height: int, width: int, device: torch.device,
) -> torch.Tensor:
    """Build absolute positions [B, 3, T, 2] for a video block."""
    t = torch.arange(start_frame, start_frame + num_frames, device=device)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    gt, gh, gw = torch.meshgrid(t, h, w, indexing="ij")
    starts = torch.stack([gt.reshape(-1), gh.reshape(-1), gw.reshape(-1)], dim=0).unsqueeze(-1)
    ends = starts + 1
    pos = torch.cat([starts, ends], dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1)


def build_audio_block_positions(
    batch_size: int, start_frame: int, num_frames: int,
    audio_tokens_per_frame: int, device: torch.device,
) -> torch.Tensor:
    """Build absolute positions [B, 1, T, 2] for an audio block."""
    t = torch.arange(start_frame, start_frame + num_frames, device=device)
    t = t.repeat_interleave(audio_tokens_per_frame)
    starts = t.view(1, -1, 1)
    ends = starts + 1
    pos = torch.cat([starts, ends], dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1)


def build_audio_window_positions(
    batch_size: int, num_window_frames: int,
    audio_tokens_per_frame: int, device: torch.device,
) -> torch.Tensor:
    """Build relative window positions [B, 1, T, 2] for audio KV cache."""
    t = torch.arange(num_window_frames, device=device).repeat_interleave(audio_tokens_per_frame)
    starts = t.view(1, -1, 1)
    ends = starts + 1
    pos = torch.cat([starts, ends], dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1)


# ---------------------------------------------------------------------------
# KV cache save/restore
# ---------------------------------------------------------------------------

def save_cache_state(caches: list[dict]) -> list[dict]:
    """Snapshot the mutable indices of every layer's KV cache."""
    return [
        {
            "global_end_index": c["global_end_index"].clone(),
            "local_end_index": c["local_end_index"].clone(),
        }
        for c in caches
    ]


def restore_cache_state(caches: list[dict], saved: list[dict]) -> None:
    """Rewind cache indices so a block can be re-generated from scratch."""
    for c, s in zip(caches, saved):
        c["global_end_index"].copy_(s["global_end_index"])
        c["local_end_index"].copy_(s["local_end_index"])


# ---------------------------------------------------------------------------
# Convert pretrained LTXModel → CausalLTXModel
# ---------------------------------------------------------------------------

def load_causal_model(
    model_ledger: ModelLedger,
    local_attn_size: int,
    frames_per_block_latent: int,
    device: torch.device,
) -> CausalLTXModel:
    """Load pretrained weights into a CausalLTXModel.

    Strategy: build the normal X0Model via ModelLedger, extract the
    velocity_model state dict + config, create CausalLTXModel with the
    same config + causal params, load state dict.  Weight names are
    identical between LTXModel and CausalLTXModel by design.
    """
    logger.info("Loading pretrained transformer (non-causal) to extract weights...")
    x0_model = model_ledger.transformer()
    velocity_model = x0_model.velocity_model
    state_dict = velocity_model.state_dict()

    # Read config from checkpoint metadata
    config = model_ledger.transformer_builder.model_config()
    t_cfg = config.get("transformer", {})

    logger.info("Creating CausalLTXModel and loading weights...")
    causal_model = CausalLTXModel(
        model_type=t_cfg.get("model_type", "AudioVideo"),
        num_attention_heads=t_cfg.get("num_attention_heads", 32),
        attention_head_dim=t_cfg.get("attention_head_dim", 128),
        in_channels=t_cfg.get("in_channels", 128),
        out_channels=t_cfg.get("out_channels", 128),
        num_layers=t_cfg.get("num_layers", 48),
        cross_attention_dim=t_cfg.get("cross_attention_dim", 4096),
        norm_eps=t_cfg.get("norm_eps", 1e-06),
        positional_embedding_theta=t_cfg.get("positional_embedding_theta", 10000.0),
        positional_embedding_max_pos=t_cfg.get("positional_embedding_max_pos", [20, 2048, 2048]),
        timestep_scale_multiplier=t_cfg.get("timestep_scale_multiplier", 1000),
        use_middle_indices_grid=t_cfg.get("use_middle_indices_grid", True),
        audio_num_attention_heads=t_cfg.get("audio_num_attention_heads", 32),
        audio_attention_head_dim=t_cfg.get("audio_attention_head_dim", 64),
        audio_in_channels=t_cfg.get("audio_in_channels", 128),
        audio_out_channels=t_cfg.get("audio_out_channels", 128),
        audio_cross_attention_dim=t_cfg.get("audio_cross_attention_dim", 2048),
        audio_positional_embedding_max_pos=t_cfg.get("audio_positional_embedding_max_pos", [20]),
        av_ca_timestep_scale_multiplier=t_cfg.get("av_ca_timestep_scale_multiplier", 1),
        cross_attention_adaln=t_cfg.get("cross_attention_adaln", False),
        # Causal-specific
        local_attn_size=local_attn_size,
        sink_size=1,
        num_frame_per_block=frames_per_block_latent,
    )

    # Load weights (names are identical between LTXModel and CausalLTXModel)
    missing, unexpected = causal_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading causal model: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys when loading causal model: %s", unexpected[:10])

    # Free the non-causal model
    del x0_model, velocity_model, state_dict
    cleanup_memory()

    return causal_model.to(device=device, dtype=torch.bfloat16).eval()


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_streaming_generation(
    model_ledger: ModelLedger,
    initial_prompt: str,
    negative_prompt: str,
    action_prompts: list[str],
    action_at_block: list[int],
    num_blocks: int,
    frames_per_block: int,
    height: int,
    width: int,
    frame_rate: float,
    num_inference_steps: int,
    local_attn_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], Audio]:
    """Generate video+audio block-by-block with causal KV-cached streaming.

    Returns decoded video chunks and audio for final assembly.
    """
    dtype = torch.bfloat16
    bsz = 1

    # ------------------------------------------------------------------
    # 1. Encode prompts
    # ------------------------------------------------------------------
    all_prompts = [initial_prompt, negative_prompt] + action_prompts
    logger.info("Encoding %d prompts through text encoder...", len(all_prompts))
    encoded = encode_prompts(all_prompts, model_ledger)

    v_ctx_pos = encoded[0].video_encoding  # [1, seq, dim]
    a_ctx_pos = encoded[0].audio_encoding
    v_ctx_neg = encoded[1].video_encoding
    a_ctx_neg = encoded[1].audio_encoding

    action_ctx_map: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, blk_idx in enumerate(action_at_block):
        enc = encoded[2 + i]
        action_ctx_map[blk_idx] = (enc.video_encoding, enc.audio_encoding)

    # ------------------------------------------------------------------
    # 2. Compute latent dimensions
    # ------------------------------------------------------------------
    # Latent spatial dims
    v_h = height // VIDEO_SCALE_FACTORS.height    # e.g. 512/32 = 16
    v_w = width // VIDEO_SCALE_FACTORS.width      # e.g. 768/32 = 24
    tpf_video = v_h * v_w                          # tokens per video frame

    # Latent frames per block
    lf_per_block = max(1, frames_per_block // VIDEO_SCALE_FACTORS.time)

    # Audio tokens per latent frame
    # AudioLatentShape has mel_bins=16 but after patchification (patch_size=1),
    # each latent frame = 1 token.  The audio latent is [B, channels=8, frames, mel_bins=16].
    # After patchification: [B, frames, channels*mel_bins] = [B, frames, 128].
    # So audio_tokens_per_frame = 1 (one token per latent frame).
    # But we need to know how many audio latent frames per video latent frame.
    block_pixel_frames = lf_per_block * VIDEO_SCALE_FACTORS.time
    block_duration = block_pixel_frames / frame_rate
    block_a_shape = AudioLatentShape.from_duration(batch=1, duration=block_duration)
    audio_latent_frames_per_block = block_a_shape.frames
    tpf_audio = max(1, audio_latent_frames_per_block // lf_per_block)  # audio tokens per video frame

    c_video = VIDEO_LATENT_CHANNELS  # 128
    c_audio = block_a_shape.channels * block_a_shape.mel_bins  # 8 * 16 = 128

    total_latent_frames = num_blocks * lf_per_block
    actual_audio_tokens_per_block = lf_per_block * tpf_audio
    total_audio_latent_frames = num_blocks * actual_audio_tokens_per_block

    logger.info(
        "Latent dims: video=[%d, %d, %d, %d] per block, audio=%d frames/block, "
        "tpf_video=%d, tpf_audio=%d",
        c_video, lf_per_block, v_h, v_w,
        audio_latent_frames_per_block, tpf_video, tpf_audio,
    )

    # ------------------------------------------------------------------
    # 3. Load causal model
    # ------------------------------------------------------------------
    causal_model = load_causal_model(
        model_ledger, local_attn_size, lf_per_block, device,
    )

    # ------------------------------------------------------------------
    # 4. Initialize KV caches
    # ------------------------------------------------------------------
    max_v_tokens = local_attn_size * tpf_video
    max_a_tokens = local_attn_size * tpf_audio
    v_kv = causal_model.init_video_kv_cache(bsz, max_v_tokens, dtype, device)
    a_kv = causal_model.init_audio_kv_cache(bsz, max_a_tokens, dtype, device)
    # No cross-attention cache — context changes per action, so we re-compute.
    v_ca = None
    a_ca = None

    # ------------------------------------------------------------------
    # 5. Prepare diffusion schedule and noise generator
    # ------------------------------------------------------------------
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(
        dtype=torch.float32, device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed)

    # ------------------------------------------------------------------
    # 6. Accumulation buffers
    # ------------------------------------------------------------------
    full_video_latent = torch.zeros(
        c_video, total_latent_frames, v_h, v_w, device=device, dtype=dtype,
    )
    full_audio_latent = torch.zeros(
        block_a_shape.channels, total_audio_latent_frames, block_a_shape.mel_bins,
        device=device, dtype=dtype,
    )

    # ------------------------------------------------------------------
    # 7. Block-by-block streaming loop
    # ------------------------------------------------------------------
    logger.info(
        "Starting causal streaming: %d blocks, %d latent frames/block, "
        "%d denoising steps/block",
        num_blocks, lf_per_block, num_inference_steps,
    )
    block_times = []
    current_frame = 0  # running latent frame counter

    for bi in range(num_blocks):
        t0 = time.perf_counter()

        # Switch action context if scheduled
        if bi in action_ctx_map:
            v_ctx_pos, a_ctx_pos = action_ctx_map[bi]
            logger.info("Block %d: ACTION SWITCH → new prompt", bi)

        n_frames = lf_per_block  # latent frames this block

        # --- Build positions for this block ---
        v_pos = build_video_block_positions(bsz, current_frame, n_frames, v_h, v_w, device)
        a_pos = build_audio_block_positions(
            bsz, current_frame, n_frames, tpf_audio, device,
        )

        # Window positions (relative, for KV-cache RoPE)
        window_frames = min(current_frame + n_frames, local_attn_size)
        v_win = build_window_positions(window_frames, v_h, v_w, bsz, device, with_bounds=True)
        a_win = build_audio_window_positions(bsz, window_frames, tpf_audio, device)

        # --- Sample noise for this block ---
        v_noise = torch.randn(
            bsz, n_frames * tpf_video, c_video,
            generator=generator, device=device, dtype=dtype,
        )
        a_noise = torch.randn(
            bsz, n_frames * tpf_audio, c_audio,
            generator=generator, device=device, dtype=dtype,
        )

        # --- Save KV cache state (rewind between denoising steps) ---
        saved_v = save_cache_state(v_kv)
        saved_a = save_cache_state(a_kv)

        # --- Multi-step Euler denoising ---
        v_latent = v_noise * sigmas[0]  # scale noise by initial sigma
        a_latent = a_noise * sigmas[0]

        for step_idx in range(len(sigmas) - 1):
            sigma_cur = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # Rewind cache to pre-block state (discard this block's K/V)
            restore_cache_state(v_kv, saved_v)
            restore_cache_state(a_kv, saved_a)

            v_steps = torch.full(
                (bsz, n_frames * tpf_video), sigma_cur.item(),
                device=device, dtype=dtype,
            )
            a_steps = torch.full(
                (bsz, n_frames * tpf_audio), sigma_cur.item(),
                device=device, dtype=dtype,
            )
            sigma_scalar = torch.full((bsz,), sigma_cur.item(), device=device, dtype=dtype)

            v_mod = Modality(
                latent=v_latent, sigma=sigma_scalar, timesteps=v_steps,
                positions=v_pos, context=v_ctx_pos, enabled=True,
            )
            a_mod = Modality(
                latent=a_latent, sigma=sigma_scalar, timesteps=a_steps,
                positions=a_pos, context=a_ctx_pos, enabled=True,
            )

            # Forward through causal model (appends K/V to cache)
            vx, ax = causal_model(
                video=v_mod, audio=a_mod,
                perturbations=None,
                kv_cache=v_kv, crossattn_cache=v_ca,
                audio_kv_cache=a_kv, audio_crossattn_cache=a_ca,
                current_start=current_frame * tpf_video,
                tokens_per_frame=tpf_video,
                window_positions=v_win,
                audio_window_positions=a_win,
            )

            # Velocity → x0
            v_x0 = to_denoised(v_latent, vx, sigma_cur)
            a_x0 = to_denoised(a_latent, ax, sigma_cur)

            # Euler step: x_{t-1} = x0 + sigma_next * velocity
            dt = sigma_next - sigma_cur
            v_vel = (v_latent.float() - v_x0.float()) / sigma_cur.float()
            a_vel = (a_latent.float() - a_x0.float()) / sigma_cur.float()
            v_latent = (v_latent.float() + v_vel * dt).to(dtype)
            a_latent = (a_latent.float() + a_vel * dt).to(dtype)

        # After the last step, the cache retains K/V from the final forward
        # pass (most denoised state). This is intentional — future blocks
        # attend to these cached representations.

        # --- Store denoised block into full buffer ---
        # Reshape flat tokens back to spatial: [B, F*H*W, C] → [C, F, H, W]
        v_block = v_latent[0].reshape(n_frames, v_h, v_w, c_video)
        v_block = v_block.permute(3, 0, 1, 2)  # [C, F, H, W]
        lf_start = bi * lf_per_block
        full_video_latent[:, lf_start:lf_start + n_frames] = v_block

        # Audio: [B, n_audio_tokens, c_audio] → [channels, n_audio_tokens, mel_bins]
        n_audio_tokens = a_latent.shape[1]
        a_block = a_latent[0].reshape(n_audio_tokens, block_a_shape.channels, block_a_shape.mel_bins)
        a_block = a_block.permute(1, 0, 2)  # [channels, n_audio_tokens, mel_bins]
        af_start = bi * n_audio_tokens
        full_audio_latent[:, af_start:af_start + n_audio_tokens] = a_block

        current_frame += n_frames

        dt = time.perf_counter() - t0
        block_times.append(dt)
        logger.info(
            "Block %d/%d done in %.2fs | cum_frames=%d",
            bi + 1, num_blocks, dt, current_frame,
        )

    # ------------------------------------------------------------------
    # 8. Free transformer, decode with VAE
    # ------------------------------------------------------------------
    del causal_model, v_kv, a_kv, v_ca, a_ca
    cleanup_memory()

    # Move latents to CPU to free VRAM for decoders
    full_video_latent_cpu = full_video_latent.cpu()
    full_audio_latent_cpu = full_audio_latent.cpu()
    del full_video_latent, full_audio_latent
    cleanup_memory()

    logger.info("Decoding video latent → pixels...")
    video_decoder = model_ledger.video_decoder()
    vae_tiling = TilingConfig(temporal_config=TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8))
    # Eagerly decode all tiles under inference_mode to avoid saving autograd intermediates
    with torch.inference_mode():
        decoded_chunks = list(vae_decode_video(
            full_video_latent_cpu.unsqueeze(0).to(device), video_decoder,
            tiling_config=vae_tiling, generator=generator,
        ))
    del video_decoder, full_video_latent_cpu
    cleanup_memory()
    decoded_video = iter(decoded_chunks)

    logger.info("Decoding audio latent → waveform...")
    audio_decoder = model_ledger.audio_decoder()
    vocoder = model_ledger.vocoder()
    with torch.inference_mode():
        decoded_audio = vae_decode_audio(full_audio_latent_cpu.unsqueeze(0).to(device), audio_decoder, vocoder)
    del audio_decoder, vocoder, full_audio_latent_cpu
    cleanup_memory()

    # Print stats
    total_wall = sum(block_times)
    total_pixel_frames = total_latent_frames * VIDEO_SCALE_FACTORS.time
    gen_seconds = total_pixel_frames / frame_rate
    rtf = gen_seconds / total_wall if total_wall > 0 else 0
    logger.info("=" * 60)
    logger.info("Streaming generation complete")
    logger.info("  blocks=%d, latent_frames=%d, pixel_frames≈%d",
                num_blocks, total_latent_frames, total_pixel_frames)
    logger.info("  generated_seconds=%.1f, wall_seconds=%.2f", gen_seconds, total_wall)
    logger.info("  realtime_factor=%.3fx", rtf)
    logger.info("  avg_block_ms=%.1f, p95_block_ms=%.1f",
                sum(block_times) / len(block_times) * 1000,
                sorted(block_times)[int(0.95 * (len(block_times) - 1))] * 1000)
    logger.info("=" * 60)

    return decoded_video, decoded_audio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal streaming AV generation with action conditioning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Prompts
    parser.add_argument("--initial-prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted")
    parser.add_argument("--action-prompts", nargs="*", default=[])
    parser.add_argument("--action-at-block", nargs="*", type=int, default=[])

    # Generation
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--frames-per-block", type=int, default=16,
                        help="Pixel frames per block (divisible by temporal scale=8).")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--local-attn-size", type=int, default=32,
                        help="KV cache window size in latent frames.")

    # Output
    parser.add_argument("--output-path", default="output_stream.mp4")

    args = parser.parse_args()

    if len(args.action_prompts) != len(args.action_at_block):
        parser.error("--action-prompts and --action-at-block must have the same number of entries.")

    device = torch.device(args.device)

    logger.info("Initializing model ledger...")
    model_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.checkpoint_path,
        gemma_root_path=args.gemma_root,
    )

    decoded_video, decoded_audio = run_streaming_generation(
        model_ledger=model_ledger,
        initial_prompt=args.initial_prompt,
        negative_prompt=args.negative_prompt,
        action_prompts=args.action_prompts,
        action_at_block=args.action_at_block,
        num_blocks=args.num_blocks,
        frames_per_block=args.frames_per_block,
        height=args.height,
        width=args.width,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        local_attn_size=args.local_attn_size,
        seed=args.seed,
        device=device,
    )

    logger.info("Writing output to %s ...", args.output_path)
    encode_video(
        video=decoded_video,
        fps=args.frame_rate,
        audio=decoded_audio,
        output_path=args.output_path,
        video_chunks_number=1,
    )
    logger.info("Done! Saved to %s", args.output_path)


if __name__ == "__main__":
    main()
