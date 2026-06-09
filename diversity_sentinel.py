# diversity_sentinel.py
# DiversitySentinel v2.2 — Standalone anti-mode-collapse module
#
# Operates alongside GravBalancer (v10.8+) without modifying it.
# GravBalancer manages learning rates (outer loop).
# DiversitySentinel adds a diversity loss to the Generator (gradient level).
#
# Key design: the two systems work on ORTHOGONAL layers:
#   - GravBalancer: adjusts HOW FAST players learn (LR)
#   - DiversitySentinel: adjusts WHAT the Generator optimizes (loss)
#
# Integration contract:
#   1. Sentinel's diversity loss is added to G's loss BEFORE backward()
#   2. GravBalancer's proxies are computed from the MAIN adversarial loss only
#      (not polluted by diversity loss) to prevent feedback loops.
#      NOTE (GravBalancer v10.9+): proxies must be NON-NEGATIVE — wrap
#      sign-changing losses with softplus, not relu (relu collapses signal).
#   3. Sentinel has its own PID for lambda — independent of GravBalancer
#   4. GravBalancer is NEVER modified — Sentinel is a pure add-on
#
# Version history:
#   v2.2: Repair release (docs/repair_plan_v1_0.md tier 1):
#         - cfg default is no longer a mutable class-level instance
#           (was: single shared DiversitySentinelConfig across all
#           default-constructed sentinels)
#         - compute() validates fake/real batch sizes match (sorted 1D
#           Wasserstein silently requires equal lengths)
#         - file re-encoded to clean UTF-8 (was mojibake)
#         - header version/compat references fixed (was: v2.0 / v10.6.8)
#   v2.1: tripwire (distress level 1) no longer triggers lambda backoff;
#         only shock/mech/panic (level >= 2).
#   v2.0: Adapted from RobustDiversitySentinel v8.2 (GravGAN notebook):
#         removed nn.Module dependency, added GravBalancer bridge,
#         diagnostics dict, configurable projector, pause_lambda.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class DiversitySentinelConfig:
    """All Sentinel parameters. Sane defaults for typical GAN training."""

    # ── Metric weights ──
    w_radial: float = 1.0           # Radial Wasserstein (density spread)
    w_sliced: float = 2.0           # Sliced Wasserstein (angular coverage)

    # ── Sliced Wasserstein ──
    n_projections: int = 32         # random projection axes per step

    # ── PID controller for lambda ──
    use_pid: bool = True
    pid_target_ratio: float = 0.15  # target: div_loss / (main_loss + div_loss)
    pid_eta: float = 0.05           # PID step size (log-space)
    pid_ema: float = 0.95           # smoothing for PID averages

    # ── Lambda limits ──
    lambda_min: float = 0.5
    lambda_max: float = 50.0
    lambda_init: float = 0.0        # start at 0 during warmup

    # ── Lambda governor (rate limiter) ──
    # Prevents lambda from jumping too fast, reducing plant noise.
    # Works like GravBalancer's rate governor — per-step max change.
    lambda_gov_enable: bool = True
    lambda_gov_up: float = 1.15     # max multiplier per step (λ_new ≤ λ_old × up)
    lambda_gov_down: float = 0.87   # min multiplier per step (λ_new ≥ λ_old × down)
    lambda_smooth_alpha: float = 0.1  # EMA smoothing on lambda output (0=no smooth, 1=instant)

    # ── Dynamics ──
    warmup_steps: int = 1000        # steps before lambda activates
    ema_beta: float = 0.99          # EMA decay for running stats

    # ── Batch scaling ──
    min_batch: int = 8              # below this → lambda = 0
    full_batch: int = 64            # at this batch size → full lambda

    # ── GravBalancer coordination ──
    respect_grav_calm: bool = True  # reduce lambda when GravBalancer is in calm
    calm_lambda_mult: float = 0.3   # multiply lambda by this during calm

    # ── Projector settings ──
    projector_dim: int = 64         # output channels for conv projector
    projector_spatial: int = 8      # adaptive pool target size
    projector_seed: int = 42        # deterministic init

    # ── RNG ──
    slice_rng_seed: int = 1337      # CPU RNG for slice directions


# ───────────────────────────────────────────────────────────────────────────
# Default projector factory
# ───────────────────────────────────────────────────────────────────────────

def make_default_projector(
    in_channels: int = 3,
    out_channels: int = 64,
    spatial: int = 8,
    seed: int = 42,
) -> nn.Module:
    """Frozen conv1x1 projector — maps images to flat feature vectors.

    For non-image data or custom feature extractors, pass your own
    projector_fn to DiversitySentinel.
    """
    proj = nn.Sequential(
        nn.AdaptiveAvgPool2d((spatial, spatial)),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    )
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        for p in proj.parameters():
            p.requires_grad = False
            if p.dim() > 1:
                nn.init.orthogonal_(p)
    finally:
        torch.set_rng_state(rng_state)
    return proj


# ───────────────────────────────────────────────────────────────────────────
# DiversitySentinel
# ───────────────────────────────────────────────────────────────────────────

class DiversitySentinel:
    """
    Anti-mode-collapse module for GAN generators.

    Works alongside GravBalancer without interfering:
    - GravBalancer manages LR balance (outer loop, no gradients)
    - Sentinel adds diversity loss to G's objective (gradient level)

    Usage in training loop:
        sentinel = DiversitySentinel(device='cuda')

        # In G step:
        loss_G_main = -D(fake).mean()                       # adversarial loss
        loss_div, div_info = sentinel.compute(fake, real)    # diversity loss
        loss_G_total = loss_G_main + loss_div                # combined
        loss_G_total.backward()
        opt_G.step()

        # After G step:
        sentinel.step(loss_G_main.item(), batch_size)

        # For GravBalancer — use MAIN loss only as proxy, non-negative
        # (GravBalancer v10.9 raises on negative proxies; use softplus):
        proxy_g = F.softplus(-d_fake).mean().item()         # NOT loss_G_total!
        lrs = grav_balancer.adjust(proxy_d, proxy_g)

        # Optional: tell sentinel about GravBalancer state
        sentinel.notify_grav_state(grav_balancer.last_metrics)
    """

    def __init__(
        self,
        cfg: Optional[DiversitySentinelConfig] = None,
        device: str = "cpu",
        projector_fn: Optional[Callable[[], nn.Module]] = None,
        in_channels: int = 3,
    ):
        # (v2.2) cfg default was a mutable class-level instance shared by
        # every default-constructed sentinel; now created per instance.
        self.cfg = cfg if cfg is not None else DiversitySentinelConfig()
        cfg = self.cfg
        self.device = torch.device(device)

        # ── Lambda & step counter ──
        self.lambda_div: float = cfg.lambda_init
        self.step_counter: int = 0

        # ── PID state ──
        self.pid_main_avg: float = 0.0
        self.pid_div_avg: float = 0.0
        self.pid_initialized: bool = False

        # ── Running statistics ──
        self.raw_error_ema: float = 0.0
        self.ema_loss_G: float = 1.0
        self._stats_initialized: bool = False

        # ── Center of mass (lazy init on first forward) ──
        self._center_real_ema: Optional[torch.Tensor] = None

        # ── Debug ──
        self._batch_factor: float = 1.0

        # ── Lambda governor state ──
        self._lambda_prev: float = cfg.lambda_min  # previous governed lambda
        self._lambda_smooth: float = 0.0           # EMA-smoothed output
        self._lambda_raw_pid: float = 0.0          # diagnostics

        # ── GravBalancer coordination ──
        self._grav_calm: bool = False
        self._grav_calm_cause: int = 0
        self._grav_distress_level: int = 0
        self._grav_distress_kind: str = "none"

        # ── Diagnostics ──
        self.last_info: Dict[str, float] = {}

        # ── Projector ──
        if projector_fn is not None:
            self.projector = projector_fn().to(self.device)
        else:
            self.projector = make_default_projector(
                in_channels=in_channels,
                out_channels=cfg.projector_dim,
                spatial=cfg.projector_spatial,
                seed=cfg.projector_seed,
            ).to(self.device)

        # Freeze projector
        for p in self.projector.parameters():
            p.requires_grad = False

        # ── Dedicated CPU RNG for slice directions ──
        self._proj_rng = torch.Generator(device="cpu")
        self._proj_rng.manual_seed(cfg.slice_rng_seed)

    # ───────────────────────────────────────────────────────── Features

    def get_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract flat feature vectors. Handles images and raw vectors."""
        if img.ndim == 4:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            if img.shape[1] == 3:
                return self.projector(img).reshape(img.size(0), -1)
            else:
                return img.reshape(img.size(0), -1)
        else:
            return img.reshape(img.size(0), -1)

    # ───────────────────────────────────────────────────────── Core math

    def _get_slice_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        """Random projection matrix (CPU-safe RNG → device)."""
        mat = torch.randn(dim, self.cfg.n_projections, device="cpu",
                          generator=self._proj_rng)
        mat = mat.to(device)
        return F.normalize(mat, p=2, dim=0)

    def _update_center_ema(self, flat_real: torch.Tensor) -> None:
        """Lazy-init + EMA update of real data centroid."""
        current_mean = flat_real.mean(dim=0).detach()

        if self._center_real_ema is None:
            self._center_real_ema = current_mean.clone()
            return

        if self._center_real_ema.numel() != current_mean.numel():
            raise RuntimeError(
                f"Center dim mismatch: {self._center_real_ema.numel()} vs "
                f"{current_mean.numel()}"
            )

        beta = 0.999
        self._center_real_ema = (
            self._center_real_ema * beta + current_mean * (1 - beta)
        )

    def _compute_losses(
        self, flat_fake: torch.Tensor, flat_real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss_radial:  weighted radial Wasserstein
            raw_radial:   unweighted radial (for EMA tracking)
            loss_sliced:  sliced Wasserstein
            r_fake:       log-radii of fake samples (for spread diagnostic)
        """
        # Center update
        with torch.no_grad():
            self._update_center_ema(flat_real)

        center = self._center_real_ema
        fake_c = flat_fake - center
        real_c = flat_real - center

        # ── Radial Wasserstein (density spread) ──
        r_fake = torch.log(torch.norm(fake_c, p=2, dim=1) + 1e-6)
        r_real = torch.log(torch.norm(real_c, p=2, dim=1) + 1e-6)

        r_fake_s, _ = torch.sort(r_fake)
        r_real_s, _ = torch.sort(r_real)

        delta_rad = torch.abs(r_fake_s - r_real_s)

        with torch.no_grad():
            w_rad = 1.0 - torch.exp(-2.0 * delta_rad)

        loss_radial = (w_rad * delta_rad).mean()
        raw_radial = delta_rad.mean()

        # ── Sliced Wasserstein (angular coverage) ──
        slice_mat = self._get_slice_matrix(fake_c.shape[1], fake_c.device)

        proj_fake = torch.mm(fake_c, slice_mat)
        proj_real = torch.mm(real_c, slice_mat)

        p_fake_s, _ = torch.sort(proj_fake, dim=0)
        p_real_s, _ = torch.sort(proj_real, dim=0)

        loss_sliced = torch.abs(p_fake_s - p_real_s).mean()

        return loss_radial, raw_radial, loss_sliced, r_fake

    # ───────────────────────────────────────────────────────── EMA utils

    def _ema_update(self, attr: str, value: float) -> None:
        beta = self.cfg.ema_beta
        old = getattr(self, attr)
        if not self._stats_initialized:
            setattr(self, attr, value)
        else:
            setattr(self, attr, old * beta + value * (1 - beta))

    # ───────────────────────────────────────────────────────── PID

    def _update_pid(self, loss_main: float, raw_error: float) -> float:
        """Log-space PID for lambda — keeps div/main ratio near target."""
        cfg = self.cfg

        if not self.pid_initialized:
            self.pid_main_avg = loss_main
            self.pid_div_avg = raw_error
            self.pid_initialized = True
            return self.lambda_div

        a = cfg.pid_ema
        self.pid_main_avg = self.pid_main_avg * a + loss_main * (1 - a)
        self.pid_div_avg = self.pid_div_avg * a + raw_error * (1 - a)

        eps = 1e-9
        main_avg = self.pid_main_avg
        div_avg = self.pid_div_avg

        if abs(main_avg) < eps and abs(div_avg) < eps:
            return self.lambda_div

        current_lam = max(self.lambda_div, cfg.lambda_min)
        term_div = current_lam * div_avg
        p = term_div / (abs(main_avg) + term_div + eps)
        err = p - cfg.pid_target_ratio
        log_lam = math.log(current_lam + eps) - cfg.pid_eta * err
        return math.exp(log_lam)

    # ───────────────────────────────────────────────────────── Lambda governor

    def _govern_lambda(self, raw_lambda: float) -> float:
        """
        Rate-limit lambda changes — prevents sharp jumps that inject
        noise into the plant (regardless of whether GravBalancer exists).

        Pipeline:
            PID output → clamp(min, max) → governor(rate limit) → EMA smooth → out

        Analogous to GravBalancer's rate governor for LR, but for lambda.
        """
        cfg = self.cfg

        if not cfg.lambda_gov_enable:
            self._lambda_prev = raw_lambda
            self._lambda_smooth = raw_lambda
            return raw_lambda

        prev = self._lambda_prev
        eps = 1e-12

        # Rate clamp: λ_new ∈ [prev × down, prev × up]
        if prev > eps:
            floor = prev * cfg.lambda_gov_down
            ceil = prev * cfg.lambda_gov_up
            governed = max(floor, min(raw_lambda, ceil))
        else:
            # First step after warmup — no history to rate-limit from.
            # Allow jump to PID target, but cap at gentle initial value.
            governed = min(raw_lambda, cfg.lambda_min * cfg.lambda_gov_up)

        # EMA smoothing on output (extra anti-jitter)
        alpha = cfg.lambda_smooth_alpha
        smoothed = self._lambda_smooth * (1 - alpha) + governed * alpha

        self._lambda_prev = governed
        self._lambda_smooth = smoothed

        return smoothed

    # ───────────────────────────────────────────────────────── Public API

    def compute(
        self,
        fake: torch.Tensor,
        real: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute diversity loss to add to Generator's loss.

        Call this BEFORE loss_G_total.backward().

        Args:
            fake: Generator output (requires grad)
            real: Real data batch (no grad needed)

        Returns:
            total_loss: scalar tensor (add to loss_G)
            info: diagnostic dict
        """
        # (v2.2) Sorted 1D Wasserstein (radial and sliced) silently requires
        # equal sample counts; a mismatch used to surface as an opaque shape
        # error (or, worse, broadcast) deep inside _compute_losses.
        if fake.size(0) != real.size(0):
            raise ValueError(
                f"DiversitySentinel.compute(): fake batch ({fake.size(0)}) and "
                f"real batch ({real.size(0)}) must have the same size — "
                "sorted 1D Wasserstein compares per-rank pairs.")

        feat_fake = self.get_features(fake)
        with torch.no_grad():
            feat_real = self.get_features(real)

        loss_rad, raw_rad, loss_slice, r_fake = self._compute_losses(
            feat_fake, feat_real
        )

        total_match = (
            loss_rad * self.cfg.w_radial + loss_slice * self.cfg.w_sliced
        )
        raw_error_total = (raw_rad + loss_slice).item()

        self._ema_update("raw_error_ema", raw_error_total)

        # Apply lambda
        effective_lambda = self.lambda_div

        # GravBalancer calm coordination
        # v2.1: only reduce lambda for real emergencies (shock/mech/panic),
        # not for tripwire (legacy calm) which is routine noise management
        if self.cfg.respect_grav_calm and self._grav_distress_level >= 2:
            effective_lambda *= self.cfg.calm_lambda_mult

        total_loss = total_match * effective_lambda

        info = {
            "div/match_error": total_match.item(),
            "div/raw_error": raw_error_total,
            "div/lambda": self.lambda_div,
            "div/lambda_effective": effective_lambda,
            "div/grav_distress_level": self._grav_distress_level,
            "div/grav_distress_kind": self._grav_distress_kind,
            "div/lambda_raw_pid": self._lambda_raw_pid,
            "div/lambda_prev": self._lambda_prev,
            "div/loss_total": total_loss.item(),
            "div/spread_fake_batch": r_fake.var().item(),
            "div/batch_factor": self._batch_factor,
            "div/grav_calm_active": float(self._grav_calm),
            "div/governor_active": float(self.cfg.lambda_gov_enable),
        }
        self.last_info = info
        return total_loss, info

    def step(self, loss_G_main: float, batch_size: int) -> None:
        """
        Post-step update: advances PID, lambda, warmup counter.

        Call AFTER opt_G.step().

        Args:
            loss_G_main: the MAIN adversarial loss value (NOT total with div)
            batch_size: current batch size
        """
        self.step_counter += 1
        self._ema_update("ema_loss_G", abs(loss_G_main))

        if not self._stats_initialized:
            self._stats_initialized = True
            return

        # Warmup — no lambda yet
        if self.step_counter < self.cfg.warmup_steps:
            self.lambda_div = 0.0
            return

        # PID update
        raw_error = self.raw_error_ema

        if self.cfg.use_pid:
            new_lambda = self._update_pid(loss_G_main, raw_error)
        else:
            new_lambda = self.cfg.lambda_min

        new_lambda = max(self.cfg.lambda_min,
                         min(new_lambda, self.cfg.lambda_max))

        # Batch scaling
        if batch_size < self.cfg.full_batch:
            denom = self.cfg.full_batch - self.cfg.min_batch
            factor = (batch_size - self.cfg.min_batch) / (denom + 1e-9)
            factor = max(0.0, min(1.0, factor))
            self._batch_factor = factor
            new_lambda *= factor
        else:
            self._batch_factor = 1.0

        # Governor: rate-limit lambda to prevent plant noise
        lambda_raw = new_lambda
        new_lambda = self._govern_lambda(new_lambda)

        self.lambda_div = new_lambda
        self._lambda_raw_pid = lambda_raw  # for diagnostics

    # ───────────────────────────────────────────────────────── GravBalancer bridge

    def notify_grav_state(self, grav_metrics: Optional[dict] = None) -> None:
        """
        Optional: inform Sentinel about GravBalancer's current state.

        This enables soft coordination:
        - During GravBalancer distress → Sentinel reduces its lambda
          (avoids injecting diversity pressure when the system is recovering)

        Args:
            grav_metrics: GravBalancer.last_metrics dict (or None to clear)
        """
        if grav_metrics is None:
            self._grav_calm = False
            self._grav_calm_cause = 0
            self._grav_distress_level = 0
            self._grav_distress_kind = "none"
            return

        self._grav_calm = bool(grav_metrics.get("calm", False))
        self._grav_calm_cause = int(grav_metrics.get("calm_cause", 0))
        self._grav_distress_level = int(grav_metrics.get("distress_level", 0))
        self._grav_distress_kind = str(grav_metrics.get("distress_kind", "none"))

    # ───────────────────────────────────────────────────────── State management

    def reset(self) -> None:
        """Full state reset for new experiment."""
        self.lambda_div = self.cfg.lambda_init
        self.step_counter = 0
        self.pid_main_avg = 0.0
        self.pid_div_avg = 0.0
        self.pid_initialized = False
        self.raw_error_ema = 0.0
        self.ema_loss_G = 1.0
        self._stats_initialized = False
        self._center_real_ema = None
        self._batch_factor = 1.0
        self._lambda_prev = self.cfg.lambda_min
        self._lambda_smooth = 0.0
        self._lambda_raw_pid = 0.0
        self._grav_calm = False
        self._grav_calm_cause = 0
        self._grav_distress_level = 0
        self._grav_distress_kind = "none"
        self.last_info = {}
        self._proj_rng.manual_seed(self.cfg.slice_rng_seed)

    def state_dict(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            "lambda_div": self.lambda_div,
            "step_counter": self.step_counter,
            "pid_main_avg": self.pid_main_avg,
            "pid_div_avg": self.pid_div_avg,
            "pid_initialized": self.pid_initialized,
            "raw_error_ema": self.raw_error_ema,
            "ema_loss_G": self.ema_loss_G,
            "_stats_initialized": self._stats_initialized,
            "_center_real_ema": (
                self._center_real_ema.cpu()
                if self._center_real_ema is not None
                else None
            ),
            "_batch_factor": self._batch_factor,
            "_lambda_prev": self._lambda_prev,
            "_lambda_smooth": self._lambda_smooth,
        }

    def load_state_dict(self, sd: Dict) -> None:
        """Restore from checkpoint."""
        self.lambda_div = sd["lambda_div"]
        self.step_counter = sd["step_counter"]
        self.pid_main_avg = sd["pid_main_avg"]
        self.pid_div_avg = sd["pid_div_avg"]
        self.pid_initialized = sd["pid_initialized"]
        self.raw_error_ema = sd["raw_error_ema"]
        self.ema_loss_G = sd["ema_loss_G"]
        self._stats_initialized = sd["_stats_initialized"]
        center = sd.get("_center_real_ema")
        if center is not None:
            self._center_real_ema = center.to(self.device)
        else:
            self._center_real_ema = None
        self._batch_factor = sd.get("_batch_factor", 1.0)
        self._lambda_prev = sd.get("_lambda_prev", self.cfg.lambda_min)
        self._lambda_smooth = sd.get("_lambda_smooth", 0.0)


# ───────────────────────────────────────────────────────────────────────────
# Convenience: make_sentinel_for_toy
# ───────────────────────────────────────────────────────────────────────────

def make_sentinel_for_toy(
    device: str = "cpu",
    warmup_steps: int = 100,
    lambda_max: float = 20.0,
    pid_target_ratio: float = 0.10,
) -> DiversitySentinel:
    """Pre-configured sentinel for 2D toy data (no images, raw vectors)."""

    def toy_projector():
        """Identity projector — toy data is already low-dim."""
        return nn.Identity()

    cfg = DiversitySentinelConfig(
        warmup_steps=warmup_steps,
        lambda_max=lambda_max,
        pid_target_ratio=pid_target_ratio,
        min_batch=8,
        full_batch=256,
    )
    return DiversitySentinel(cfg, device=device, projector_fn=toy_projector)
