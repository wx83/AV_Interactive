#!/usr/bin/env python3
"""Causal LTX-2 distillation training.

Supports two modes controlled by --use-dmd:

  --no-use-dmd (default, 2 models):
    Simple velocity distillation.  The causal student learns to match the
    frozen bidirectional teacher's velocity on the same noisy input.
    Fits on 2× H100 80GB with FSDP.

  --use-dmd (3 models):
    Full DMD distillation with an additional trainable critic (fake_score).
    Requires ≥4× H100 80GB or equivalent memory.

Uses PyTorch Lightning with FSDP FULL_SHARD (no CPU offload).
"""

import argparse
import logging
import os
import random
import sys

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import Dataset, DataLoader

# Add LTX-2 packages to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-core", "src"))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-trainer", "src"))
sys.path.insert(0, os.path.join(_ROOT, "packages", "ltx-pipelines", "src"))

from ltx_core.model.transformer.ltx_wrapper import LTXDiffusionWrapper
from ltx_core.model.transformer.model import LTXModelType
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.utils import to_denoised
from ltx_pipelines.utils import cleanup_memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

# ── Constants ───────────────────────────────────────────────────────────
VIDEO_LATENT_CHANNELS = 128
VIDEO_TEMPORAL_DOWNSCALE = 8
VIDEO_SPATIAL_DOWNSCALE = 32


def _resolve_model_type(t_cfg: dict) -> LTXModelType:
    raw = t_cfg.get("model_type", "AudioVideo")
    if isinstance(raw, LTXModelType):
        return raw
    try:
        return LTXModelType(raw)
    except ValueError:
        return LTXModelType[raw]


def _needs_caption_projection(t_cfg: dict) -> bool:
    return not t_cfg.get("caption_proj_before_connector", False) and "caption_channels" in t_cfg


def build_model_kwargs(t_cfg: dict) -> dict:
    model_type = _resolve_model_type(t_cfg)
    kwargs = dict(
        model_type=model_type,
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
    )
    if _needs_caption_projection(t_cfg):
        kwargs["caption_projection"] = create_caption_projection(t_cfg)
        if model_type.is_audio_enabled():
            kwargs["audio_caption_projection"] = create_caption_projection(t_cfg, audio=True)
    return kwargs


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class PreEncodedPromptDataset(Dataset):
    def __init__(self, encoded_prompts: list[dict], neg_ctx: dict):
        self.encoded = encoded_prompts
        self.neg_ctx = neg_ctx

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        enc = self.encoded[idx]
        return {
            "video_ctx_pos": enc["video_pos"],
            "video_ctx_neg": self.neg_ctx["video_neg"],
        }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def build_video_positions(
    batch_size: int, num_frames: int, height: int, width: int, device: torch.device,
) -> torch.Tensor:
    t = torch.arange(num_frames, device=device)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    gt, gh, gw = torch.meshgrid(t, h, w, indexing="ij")
    starts = torch.stack([gt.reshape(-1), gh.reshape(-1), gw.reshape(-1)], dim=0).unsqueeze(-1)
    ends = starts + 1
    pos = torch.cat([starts, ends], dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1, -1)


# ═══════════════════════════════════════════════════════════════════════
# Lightning Module
# ═══════════════════════════════════════════════════════════════════════

class CausalDistillModule(L.LightningModule):
    """Causal distillation with configurable mode.

    When use_dmd=False (default):
      2-model velocity distillation (student + frozen teacher).
      ~38GB bf16 → fits on 2× H100.

    When use_dmd=True:
      3-model DMD distillation (student + frozen teacher + trainable critic).
      ~57GB bf16 → needs ≥4× H100.
    """

    def __init__(
        self,
        t_cfg: dict,
        weights_path: str,
        num_frames: int = 8,
        height: int = 512,
        width: int = 768,
        num_frame_per_block: int = 2,
        local_attn_size: int = 4,
        lr_gen: float = 1e-5,
        lr_critic: float = 1e-5,
        grad_clip: float = 1.0,
        accumulate_grad_batches: int = 1,
        gradient_checkpointing: bool = False,
        use_dmd: bool = False,
        real_guidance_scale: float = 4.5,
        fake_guidance_scale: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["t_cfg"])
        if use_dmd:
            self.automatic_optimization = False

        self._t_cfg = t_cfg

        self.v_h = height // VIDEO_SPATIAL_DOWNSCALE
        self.v_w = width // VIDEO_SPATIAL_DOWNSCALE
        self.tpf_video = self.v_h * self.v_w
        self.num_frames = num_frames
        self.total_tokens = num_frames * self.tpf_video
        self.in_channels = t_cfg.get("in_channels", 128)

        self._positions = None

    def configure_model(self):
        if hasattr(self, "gen"):
            return

        import gc
        t_cfg = self._t_cfg

        loaded = torch.load(
            self.hparams.weights_path, map_location="cpu",
            mmap=True, weights_only=False,
        )
        state_dict = loaded["state_dict"]
        del loaded

        # ── student (causal, trainable) ──
        logger.info("Creating student (causal)...")
        gen_kwargs = build_model_kwargs(t_cfg)
        self.gen = LTXDiffusionWrapper(
            is_causal=True,
            num_frame_per_block=self.hparams.num_frame_per_block,
            local_attn_size=self.hparams.local_attn_size,
            sink_size=1,
            model_kwargs=gen_kwargs,
        )
        missing, unexpected = self.gen.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Student missing %d keys (first 5): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Student unexpected %d keys (first 5): %s", len(unexpected), unexpected[:5])
        gc.collect()

        # ── teacher (bidirectional, frozen) ──
        # Stored outside nn.Module hierarchy (plain list) so FSDP does not
        # wrap it.  With bf16-true precision the student FSDP footprint is
        # small enough (~9.4 GB) to keep the teacher on GPU permanently,
        # avoiding the costly CPU↔GPU transfers that caused OOM with
        # bf16-mixed.
        logger.info("Creating teacher (bidirectional, frozen)...")
        teacher_kwargs = build_model_kwargs(t_cfg)
        teacher = LTXDiffusionWrapper(is_causal=False, model_kwargs=teacher_kwargs)
        teacher.model.load_state_dict(state_dict, strict=False)
        teacher.requires_grad_(False)
        teacher.eval()
        teacher = teacher.to(dtype=torch.bfloat16)
        self._teacher_ref = [teacher]
        gc.collect()

        # ── critic (bidirectional, trainable) — DMD mode only ──
        if self.hparams.use_dmd:
            logger.info("Creating critic (bidirectional, trainable) [DMD mode]...")
            from ltx_trainer.dmd import DMDLoss

            critic_kwargs = build_model_kwargs(t_cfg)
            self.fake_score = LTXDiffusionWrapper(is_causal=False, model_kwargs=critic_kwargs)
            self.fake_score.model.load_state_dict(state_dict, strict=False)
            gc.collect()

            if self.hparams.gradient_checkpointing:
                self.fake_score.enable_gradient_checkpointing()

            self.dmd = DMDLoss(
                generator=self.gen,
                real_score=self._teacher_ref[0],
                fake_score=self.fake_score,
                num_train_timestep=1000,
                real_guidance_scale=self.hparams.real_guidance_scale,
                fake_guidance_scale=self.hparams.fake_guidance_scale,
                num_frame_per_block=self.hparams.num_frame_per_block,
            )

        del state_dict
        gc.collect()

        if self.hparams.gradient_checkpointing:
            self.gen.enable_gradient_checkpointing()

    def _get_positions(self, batch_size: int) -> torch.Tensor:
        if self._positions is None or self._positions.shape[0] != batch_size:
            self._positions = build_video_positions(
                batch_size, self.num_frames, self.v_h, self.v_w, self.device,
            )
        return self._positions

    def configure_optimizers(self):
        gen_optim = torch.optim.AdamW(
            [p for p in self.gen.parameters() if p.requires_grad],
            lr=self.hparams.lr_gen, weight_decay=0.01,
        )
        if self.hparams.use_dmd:
            critic_optim = torch.optim.AdamW(
                [p for p in self.fake_score.parameters() if p.requires_grad],
                lr=self.hparams.lr_critic, weight_decay=0.01,
            )
            return [gen_optim, critic_optim]
        return gen_optim

    # ── Simple velocity distillation step (automatic optimization) ──
    def _velocity_step(self, batch: dict, batch_idx: int):
        B = batch["video_ctx_pos"].shape[0]
        dtype = torch.bfloat16
        positions = self._get_positions(B)
        video_ctx_pos = batch["video_ctx_pos"]

        noise = torch.randn(
            B, self.total_tokens, self.in_channels, device=self.device, dtype=dtype,
        )
        sigma_val = random.uniform(0.02, 0.98)
        timesteps = torch.full(
            (B, self.total_tokens), sigma_val, device=self.device, dtype=dtype,
        )
        sigma_tensor = torch.tensor([sigma_val], device=self.device, dtype=dtype).expand(B)

        noisy_mod = Modality(
            latent=noise,
            sigma=sigma_tensor,
            timesteps=timesteps,
            positions=positions,
            context=video_ctx_pos,
            enabled=True,
        )

        teacher = self._teacher_ref[0]
        if next(teacher.parameters()).device != self.device:
            teacher.to(self.device)
        with torch.no_grad():
            teacher_vel, _ = teacher(video=noisy_mod, audio=None)

        student_vel, _ = self.gen(video=noisy_mod, audio=None)
        loss = F.mse_loss(student_vel, teacher_vel)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ── Full DMD step ──
    def _dmd_step(self, batch: dict, batch_idx: int):
        gen_optim, critic_optim = self.optimizers()
        B = batch["video_ctx_pos"].shape[0]
        dtype = torch.bfloat16
        positions = self._get_positions(B)
        accum = self.hparams.accumulate_grad_batches
        is_accum_step = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == self.trainer.num_training_batches

        video_ctx_pos = batch["video_ctx_pos"]
        video_ctx_neg = batch["video_ctx_neg"]

        # ── Generator step ──
        if batch_idx % accum == 0:
            gen_optim.zero_grad(set_to_none=True)

        noise_latent = torch.randn(
            B, self.total_tokens, self.in_channels, device=self.device, dtype=dtype,
        )
        sigma_val = random.uniform(0.1, 0.9)
        timesteps = torch.full(
            (B, self.total_tokens), sigma_val, device=self.device, dtype=dtype,
        )

        gen_mod = Modality(
            latent=noise_latent,
            sigma=torch.tensor([sigma_val], device=self.device, dtype=dtype).expand(B),
            timesteps=timesteps,
            positions=positions,
            context=video_ctx_pos,
            enabled=True,
        )

        vx, _ = self.gen(video=gen_mod, audio=None)
        generated_video = to_denoised(noise_latent, vx, sigma_val)

        gen_loss, gen_log = self.dmd.generator_loss(
            generated_video=generated_video,
            video_positions=positions,
            video_context_pos=video_ctx_pos,
            video_context_neg=video_ctx_neg,
        )

        self.manual_backward(gen_loss / accum)

        if is_accum_step:
            self.clip_gradients(gen_optim, gradient_clip_val=self.hparams.grad_clip)
            gen_optim.step()

        # ── Critic step ──
        if batch_idx % accum == 0:
            critic_optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            noise2 = torch.randn(
                B, self.total_tokens, self.in_channels, device=self.device, dtype=dtype,
            )
            sigma_val2 = random.uniform(0.1, 0.9)
            gen_mod2 = Modality(
                latent=noise2,
                sigma=torch.tensor([sigma_val2], device=self.device, dtype=dtype).expand(B),
                timesteps=torch.full(
                    (B, self.total_tokens), sigma_val2, device=self.device, dtype=dtype,
                ),
                positions=positions,
                context=video_ctx_pos,
                enabled=True,
            )
            vx2, _ = self.gen(video=gen_mod2, audio=None)
            generated_video2 = to_denoised(noise2, vx2, sigma_val2)

        critic_loss, critic_log = self.dmd.critic_loss(
            generated_video=generated_video2,
            video_positions=positions,
            video_context_pos=video_ctx_pos,
        )

        self.manual_backward(critic_loss / accum)

        if is_accum_step:
            self.clip_gradients(critic_optim, gradient_clip_val=self.hparams.grad_clip)
            critic_optim.step()

        # ── Logging ──
        self.log("gen_loss", gen_loss, prog_bar=True, sync_dist=True)
        self.log("critic_loss", critic_loss, prog_bar=True, sync_dist=True)
        for k, v in {**gen_log, **critic_log}.items():
            self.log(k, v, sync_dist=True)

    def training_step(self, batch: dict, batch_idx: int):
        if self.hparams.use_dmd:
            self._dmd_step(batch, batch_idx)
        else:
            return self._velocity_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["gen_model_state_dict"] = {
            k: v.cpu() for k, v in self.gen.model.state_dict().items()
        }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Causal LTX-2 distillation (Lightning)")

    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--gemma-root", required=True)
    parser.add_argument("--prompt-csv", required=True)
    parser.add_argument("--prompt-column", default="Summarized Description")
    parser.add_argument("--output-dir", default="checkpoints/causal_distill")

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr-gen", type=float, default=1e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--prompt-pool-size", type=int, default=200)

    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frame-per-block", type=int, default=2)
    parser.add_argument("--local-attn-size", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # DMD mode (3 models) vs velocity distillation (2 models)
    parser.add_argument("--use-dmd", action="store_true", default=False,
                        help="Enable full DMD with critic (3 models, needs ≥4× H100)")
    parser.add_argument("--real-guidance-scale", type=float, default=4.5)
    parser.add_argument("--fake-guidance-scale", type=float, default=0.0)

    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto",
                        choices=["auto", "ddp", "fsdp"])
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["bf16-mixed", "bf16-true", "16-mixed", "32"])
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    L.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    mode_str = "DMD (3 models)" if args.use_dmd else "velocity distillation (2 models)"
    logger.info("Training mode: %s", mode_str)

    # ── Load pre-encoded prompts and weights from cache ──
    cache_path = os.path.join(args.output_dir, "encoded_prompts.pt")
    weights_cache_path = os.path.join(args.output_dir, "pretrained_weights.pt")

    if not os.path.exists(cache_path) or not os.path.exists(weights_cache_path):
        raise FileNotFoundError(
            f"Cache files not found in {args.output_dir}. "
            f"Run prepare_training_cache.py first."
        )

    logger.info("Loading cached prompts from %s", cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    all_encoded, neg_ctx = cache["encoded"], cache["neg_ctx"]
    del cache

    logger.info("Loading transformer config from %s", weights_cache_path)
    loaded = torch.load(weights_cache_path, map_location="cpu", mmap=True, weights_only=False)
    t_cfg = loaded["t_cfg"]
    del loaded

    logger.info("Transformer config: num_layers=%s, num_heads=%s, head_dim=%s, caption_proj=%s",
                t_cfg.get("num_layers"), t_cfg.get("num_attention_heads"),
                t_cfg.get("attention_head_dim"),
                "in-transformer" if _needs_caption_projection(t_cfg) else "in-encoder")

    # ── Lightning module ──
    module = CausalDistillModule(
        t_cfg=t_cfg,
        weights_path=weights_cache_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_frame_per_block=args.num_frame_per_block,
        local_attn_size=args.local_attn_size,
        lr_gen=args.lr_gen,
        lr_critic=args.lr_critic,
        grad_clip=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_checkpointing=args.gradient_checkpointing,
        use_dmd=args.use_dmd,
        real_guidance_scale=args.real_guidance_scale,
        fake_guidance_scale=args.fake_guidance_scale,
    )

    # ── Dataset + DataLoader ──
    dataset = PreEncodedPromptDataset(all_encoded, neg_ctx)
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    max_epochs = max(1, (args.num_steps + steps_per_epoch - 1) // steps_per_epoch)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Strategy ──
    if args.strategy == "fsdp":
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools

        from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
        from ltx_core.model.transformer.causal_transformer import CausalAVTransformerBlock

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={BasicAVTransformerBlock, CausalAVTransformerBlock},
        )
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy="FULL_SHARD",
            use_orig_params=True,
            limit_all_gathers=True,
        )
    elif args.strategy == "ddp":
        strategy = "ddp"
    else:
        strategy = "auto"

    # ── Callbacks ──
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="step_{step:06d}",
        every_n_train_steps=args.save_every,
        save_top_k=-1,
        save_last=True,
    )

    # ── Trainer ──
    trainer = L.Trainer(
        default_root_dir=args.output_dir,
        max_steps=args.num_steps,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=args.grad_clip if not args.use_dmd else None,
        accumulate_grad_batches=args.accumulate_grad_batches if not args.use_dmd else 1,
    )

    # ── Train ──
    trainer.fit(module, train_loader, ckpt_path=args.resume)

    # ── Save final generator weights ──
    if trainer.is_global_zero:
        gen_path = os.path.join(args.output_dir, "gen_final.pt")
        torch.save(
            {k: v.cpu() for k, v in module.gen.model.state_dict().items()},
            gen_path,
        )
        logger.info("Saved final generator weights to %s", gen_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
