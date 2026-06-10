# grav_balancer.py
# GravBalancer v10.9.1
#
# Adaptive learning rate controller for N-player optimization.
# Scenario-agnostic: works across cooperative, adversarial, and mixed regimes.
# Optimizer-agnostic: all signals derived from observable loss dynamics.
#
# ═══════════════════════════════════════════════════════════════
# Architecture
# ═══════════════════════════════════════════════════════════════
#
# Steering:
#   - Damper (u_p): tanh saturation + anti-reverse + β smoothing
#   - Integrator (u_i): off / on / auto (hysteresis gating, §6)
#   - Quiet-gate: dampens commands when system is calm (z-score bypass)
#
# Safety layers (inner → outer):
#   1. Authority: scales u_cap based on noise_norm (continuous squeeze)
#   2. Control stress: asymmetric EMA of wall/hold/sat rates → authority squeeze
#   3. Tripwire: short intervention + cooldown on sustained distress
#   4. Panic: rare brake on base_lr for systemic loss-of-control
#   5. Climate: energy-based lr_meta + anti-chatter hold for high-LR stability
#
# Pipeline (strict order):
#   stats → gaps → steering (damper+integrator) → calm →
#   authority × stress × ramp → ratio clamp → governor →
#   comfort → jump-clip → integrator update → auto-gating →
#   control stress → panic → climate → diagnostics
#
# Key design decisions:
#   - gaps_mode="dynamic" (default): steering uses baseline-removed d_filtered,
#     not level gaps. Eliminates structural offset. Level gaps retained
#     for calm/predictor/diagnostics.
#   - Anti-windup Variant C: LR walls = global stop, u_sat = conditional
#     (no push-further). Zero-sum delta_i projection reduces cross-coupling.
#   - Comfort gating on CURRENT flags (no prev-step lag).
#   - Headroom-aware comfort cap with margin prevents wall-chatter.
#   - Shock calm discharges both u_p and u_i (no spring-back).
#
# N-invariance:
#   - Panic symptoms: fractional (mean-of-players), not binary (any-player).
#     N-adaptive threshold: thr_eff = base × 2/(N+1), floored at 0.1.
#   - Warmup convergence: per-player tracking with quantile aggregation.
#     N≤2: strict (q=1.0, ok_frac=1.0). N>2: robust (q=0.8, ok_frac=0.8).
#   - Climate E: per-player resid aggregation via max (N≤2) or quantile (N>2).
#   - Shock: mean over all players (consistent with threshold calibration).
#
# Warmup: three-phase convergence-based (A → B → ramp → active).
#   Phase A: minimum observation (warmup_steps).
#   Phase B: convergence check on shock_ema dynamics per-player.
#   Ramp: quadratic soft-start of steering authority.
#   Dual-signal: mus-change for convergence, resid for snap calibration.
#
# Climate control (energy-based):
#   E = max(shock_ratio_hi, 1+wall_rate, 1+u_sat_frac, 1+clamp_frac).
#   E_fast (τ~30) catches spikes. E_slow (τ~1000) tracks chronic storms.
#   lr_meta = 1/(1 + k × max(0, E_slow - 1 - deadband)): slow LR reduction.
#   Anti-chatter hold: E_fast spike → zero steering for next step (1-step lag).
#   E tracked during ramp (pre-arm), applied only in active phase.
#
# ═══════════════════════════════════════════════════════════════
# Version history (consolidated at v10.8)
# ═══════════════════════════════════════════════════════════════
# v10.9.1: Climate fix К1 (repair_plan addendum d): winsorize the E_slow
#          INPUT at E_slow × climate_E_winsor_mult (default 3.0; ≤0 restores
#          old behavior). E_fast/hold keep raw E. Observed on toy seed-42:
#          one explosion (E=8021) at ep2.4 pinned lr_meta≈0.01..0.2 for the
#          remaining ~78 epochs (τ≈5000 steps). Storms are now measured by
#          duration, not by the amplitude of a single spike.
# v10.9:   Repair release (tiers 0-4 of docs/repair_plan_v1_0.md):
#          - Input contract: proxy >= 0; gross negatives raise ValueError,
#            tiny negatives (FP noise, within 1e-6 x scale) are clamped with
#            a one-time warning (was: silent clip of ALL negatives destroying
#            hinge/WGAN proxy signal).
#          - Warmup feasibility: convergence_window=None auto-scales from
#            warmup_steps; explicit infeasible config raises; degraded exit
#            emits warnings.warn + degraded_reason metric (was: silent
#            degraded for warmup_steps < ~400 — all v10.8.x CIFAR sweeps).
#          - reset_state(): restores dyn_deadband_z/_ramp_steps_total from
#            constructor snapshots (was: calibration leak across resets,
#            cumulative ramp doubling).
#          - Zero-sum invariant for N>2: projection onto {mean=0} ∩ box
#            (was: clip after zero-mean shifted mean(lrs) up to ~10%).
#          - Saturation measured against active-phase cap, not ramp-scaled
#            (was: soft-start artifacts polluted stress/panic/climate-E).
#          - last_metrics updated on warmup snap step (was: stale dict).
#          - Removed dead params k_stab/noise_db/stab_min (no-ops since
#            v10.7.0; loud TypeError at stale call-sites is intentional).
# v10.8:   Cleanup release. No behavioral changes from v10.7.11.
#          Fixed encoding, removed dead code, consolidated state init,
#          cleaned naming and comments, verified N-invariance.
# v10.7.11: Energy-based climate control (lr_meta + anti-chatter hold).
# v10.7.10: Fix deadband calibration regression from extended warmup.
# v10.7.9:  Convergence-based warmup + soft-start ramp.
# v10.7.8:  N-adaptive panic threshold: thr_eff = base × 2/(N+1).
# v10.7.7:  N-invariant panic symptoms (fractional, not binary).
# v10.7.6:  shock_stress_ema filter for transient spike filtering.
# v10.7.5:  Panic raw shock channel bypasses deadband.
# v10.7.4:  Dynamic deadband, intervention rate, dynamic auth_min.
# v10.7.3:  Three-layer throttle (authority/panic/tripwire), noise_ref, control stress.
# v10.6.9:  wall_ema_fast, headroom margin, comfort-chatter suppression.
# v10.6:    Adaptive rate governor.
# v10.5:    Calm/Shock/Mech detectors, flip detector, predictor.
from __future__ import annotations

import math
import warnings
from collections import deque
from typing import List, Literal, Optional, Tuple

import numpy as np


def _window_to_alpha(window: int) -> float:
    """EMA smoothing factor from window size. α = 2/(N+1)."""
    return 2.0 / (window + 1)


class GravBalancer:
    """
    GravBalancer v10.9.1

    Adaptive learning rate controller for N players.
    Manages dynamics based on observable stress proxies.
    Does NOT interpret proxy semantics -- scenario-agnostic by design.
    Input contract: proxies must be non-negative scalars (see _coerce_inputs).
    """

    def __init__(
        self,
        n_players: int = 2,
        base_lr: float = 2e-4,

        # --- EMA windows ---
        stat_window: int = 39,
        d_filter_window: int = 19,
        osc_window: int = 24,
        osc_base_window: int = 399,

        # --- Volatility floors ---
        min_volatility: float = 0.005,
        min_volatility_rel: float = 0.005,

        # --- Damper (§4.1) ---
        damper_k: float = 1.0,          # tanh steepness
        beta_u: float = 0.3,            # smoothing: u_p = (1-β)*u_p + β*targets
        reverse_damp: float = 0.3,      # anti-reverse penalty when flipping
        gap_deadband: float = 0.06,     # deadband for level-mode gaps
        gaps_mode: Literal["level", "dynamic"] = "dynamic",  # steering signal source
        # Dynamic gaps params (only used when gaps_mode="dynamic"):
        dyn_floor_min: float = 1e-4,    # min floor for d_centered normalization (raw units)
        dyn_deadband_z: float = 1.0,    # deadband in z-score units (~1σ, react to anomalies)
        dyn_deadband_z_min: float = 0.5,  # absolute min deadband (prevent too-low calibration)
        dyn_deadband_z_max: float = 3.5,  # absolute max deadband (prevent warmup drift blindness)
        dyn_deadband_z_auto: bool = True,  # auto-calibrate from warmup percentile
        dyn_deadband_z_pct: float = 0.90,  # warmup percentile for deadband calibration
        quiet_gate_enable: bool = True,    # dampen commands when system quiet
        quiet_gate_bypass_z: float = 3.0,  # bypass quiet-gate when z_max exceeds this
        quiet_gate_min: float = 0.05,      # floor: P never fully silenced

        # --- Integrator (§4.2) ---
        integrator_mode: Literal["off", "on", "auto"] = "auto",
        ki: float = 0.02,
        iterm_cap: float = 0.5,
        iterm_relax: float = 0.05,      # decay rate when held
        iterm_relax_strong: float = 0.2,  # decay rate when integrator disabled

        # --- Auto-gating (§6) ---
        sat_off_thr: float = 0.6,       # disable I if sat_ema > this
        sat_on_thr: float = 0.3,        # re-enable I if sat_ema < this
        flip_off_thr: float = 0.5,
        flip_on_thr: float = 0.25,
        hold_off_thr: float = 0.5,
        hold_on_thr: float = 0.25,
        gate_cooldown: int = 50,         # steps after disable before re-enable
        gate_ema_alpha: float = 0.05,    # EMA smoothing for gating signals

        # --- Constraints (§9) ---
        max_ratio: float = 1.618,
        u_cap: float = 0.5,

        # --- Rate governor (§10) ---
        max_jump_up: float = 1.25,
        max_jump_down: float = 0.9,
        gov_noise_sensitivity: float = 3.0,

        # --- Throttle (§8) ---
        boost_max: float = 0.1,         # max comfort boost above 1.0
        noise_boost_thr: float = 0.5,   # noise_norm below this → comfort eligible
        boost_min_duration: int = 20,   # min steps of quiet before boost kicks in
        headroom_margin: float = 0.03,  # safety margin for headroom cap
        wall_ema_fast_alpha: float = 0.15,  # fast EMA for post-comfort wall tracking
        wall_boost_thr: float = 0.25,       # block comfort if wall_ema_fast > this

        # --- noise_ref (live operational noise norm) ---
        noise_ref_window: int = 999,          # slow EMA window for noise_ref
        noise_ref_max_grow: float = 1.01,     # max per-step growth
        noise_ref_max_shrink: float = 0.99,   # max per-step shrink
        noise_ref_winsor_mult: float = 2.0,   # clip input to noise_ref * this
        noise_ref_refractory_div: float = 10.0,  # slow learning during shock/calm

        # --- Authority (Layer 1: scales u_cap) ---
        k_auth: float = 3.0,                  # steepness (replaces k_stab)
        auth_db: float = 0.3,                 # deadband (replaces noise_db)
        auth_min: float = 0.2,                # floor (replaces stab_min)

        # --- Panic throttle (Layer 2: rare brake on base_lr) ---
        panic_symptom_thr: float = 0.4,       # compound score threshold (~[0,1])
        panic_confirm_steps: int = 20,         # sustained ticks before activation
        panic_factor_min: float = 0.3,         # max brake depth
        panic_decay_rate: float = 0.02,        # descent speed per step
        panic_recover_rate: float = 0.01,      # recovery speed per step

        # --- Tripwire (short intervention + cooldown) ---
        calm_legacy_confirm: int = 5,          # N ticks to confirm legacy trigger
        trip_len: int = 10,                     # tripwire duration (short, not calm_len)
        trip_cooldown: int = 60,                # min steps between tripwires
        trip_auth_mult: float = 0.5,            # authority squeeze during tripwire

        # --- Control stress (continuous authority squeeze) ---
        stress_alpha_event: float = 0.02,       # EMA alpha for event rates (wall/hold/sat)
        stress_alpha_tighten: float = 0.05,     # fast response: squeeze authority
        stress_alpha_relax: float = 0.005,      # slow recovery: relax authority
        stress_auth_k: float = 2.0,             # stress-to-authority steepness
        stress_auth_floor: float = 0.5,         # min stress-authority factor

        # --- Calm / Shock / Mech ---
        shock_k: float = 5.0,
        snap_floor_mult: float = 3.0,
        shock_bypass_mult: float = 1.5,        # hard-bypass: shock > mult × thr
        shock_snap_floor_mult: float = 2.0,    # floor: _shock_snap ≥ mult × rel_vol_ema
        shock_sustained_enable: bool = True,    # sustained path: _shock_ema > thr
        shock_hot_mult: float = 2.0,           # comfort shock_hot: _shock_ema > mult × _shock_snap

        # --- Oscillation predictor ---
        osc_score_cap: float = 0.5,
        osc_min_baseline: float = 1e-6,
        osc_preempt_calm: bool = True,
        osc_require_high_vol: bool = True,
        osc_risk_trigger: float = 0.75,
        osc_risk_trigger_lo: Optional[float] = None,
        osc_conf_thr: float = 0.60,

        # --- Flip detector ---
        flip_win: int = 16,

        # --- Warmup ---
        warmup_steps: int = 1000,

        # --- Convergence-based warmup ---
        convergence_window: Optional[int] = None,  # steps per window; None → auto-scale from warmup_steps
        convergence_cv_thr: float = 0.3,         # CV threshold for stability
        convergence_growth_thr: float = 1.15,    # growth threshold (geo-mean log-ratio)
        convergence_n_confirm: int = 3,          # consecutive windows to confirm
        max_warmup_mult: float = 3.0,            # max_warmup = warmup_steps * mult
        ramp_i_gate: float = 0.5,               # I-term off when ramp_factor < this
        warmup_ok_frac_thr: Optional[float] = None,  # None → auto: 1.0 if N≤2, 0.8 otherwise
        warmup_q_robust: float = 0.8,           # quantile for cv/growth aggregation (N>2)
        warmup_q_snap: float = 0.9,             # quantile for snap aggregation (N>2)

        # --- Climate control (energy-based lr_meta + anti-chatter hold) ---
        climate_tau_fast: int = 30,              # E_fast EMA window (catches spikes)
        climate_tau_slow: int = 1000,            # E_slow EMA window (chronic storm)
        climate_relax_ratio_fast: float = 0.3,   # E_fast relax alpha = tighten * this
        climate_relax_ratio_slow: float = 0.1,   # E_slow relax alpha = tighten * this
        climate_k_meta: float = 1.0,             # lr_meta = 1/(1 + k * max(0, E_slow-1))
        climate_lr_meta_floor: float = 0.01,     # min lr_meta (100x reduction max)
        climate_E_hold_on: float = 2.0,          # anti-chatter hold activates
        climate_E_hold_off: float = 1.3,         # anti-chatter hold releases (hysteresis)
        climate_slew_down: float = 0.05,         # lr_meta max decrease per step
        climate_slew_up: float = 0.01,           # lr_meta max increase per step
        climate_E_deadband: float = 0.02,        # E_slow must exceed 1+this before lr_meta responds
        climate_E_winsor_mult: float = 3.0,      # (v10.9.1) clip E_slow INPUT to E_slow×this; ≤0 disables
        abs_lr_floor: Optional[float] = None,    # absolute LR floor after climate (None = base_lr*0.001)

        # --- Profile (§12) ---
        profile: Literal["competitive", "cooperative"] = "competitive",

        # --- Debug (§15) ---
        debug_freeze: Literal["none", "no_throttle", "no_integrator", "no_calm"] = "none",

    ) -> None:
        self.N = int(n_players)
        if self.N < 2:
            raise ValueError("At least 2 players required.")

        self.base_lr = float(base_lr)

        # EMA alphas
        self.stat_alpha = _window_to_alpha(stat_window)
        self.d_filter_alpha = _window_to_alpha(d_filter_window)
        self.osc_alpha = _window_to_alpha(osc_window)
        self.osc_base_alpha = _window_to_alpha(osc_base_window)

        # Volatility
        self.min_volatility = float(min_volatility)
        self.min_volatility_rel = float(min_volatility_rel)

        # Damper params
        self.damper_k = float(damper_k)
        self.beta_u = float(beta_u)
        self.reverse_damp = float(reverse_damp)
        self.gap_deadband = float(gap_deadband)
        self.gaps_mode = gaps_mode
        self.dyn_floor_min = float(dyn_floor_min)
        self.dyn_deadband_z = float(dyn_deadband_z)
        self.dyn_deadband_z_min = float(dyn_deadband_z_min)
        self.dyn_deadband_z_max = float(dyn_deadband_z_max)
        self.dyn_deadband_z_auto = bool(dyn_deadband_z_auto)
        self.dyn_deadband_z_pct = float(dyn_deadband_z_pct)
        self.dyn_deadband_z_base = float(dyn_deadband_z)  # will be overridden at end of warmup
        self.quiet_gate_enable = bool(quiet_gate_enable)
        self.quiet_gate_bypass_z = float(quiet_gate_bypass_z)
        self.quiet_gate_min = float(quiet_gate_min)
        self._quiet_factor = 1.0
        self._warmup_z_samples = []  # collect |z| during warmup for percentile calibration
        self._warmup_z_n = 0
        self._warmup_z_pct_raw = float('nan')
        self._neg_proxy_warned = False  # one-time warning for tiny negative proxies

        # Integrator params
        self.integrator_mode = integrator_mode
        self.ki = float(ki)
        self.iterm_cap = float(iterm_cap)
        self.iterm_relax = float(iterm_relax)
        self.iterm_relax_strong = float(iterm_relax_strong)

        # Auto-gating params
        self.sat_off_thr = float(sat_off_thr)
        self.sat_on_thr = float(sat_on_thr)
        self.flip_off_thr = float(flip_off_thr)
        self.flip_on_thr = float(flip_on_thr)
        self.hold_off_thr = float(hold_off_thr)
        self.hold_on_thr = float(hold_on_thr)
        self.gate_cooldown = int(gate_cooldown)
        self.gate_ema_alpha = float(gate_ema_alpha)

        # Constraints
        if max_ratio <= 1.0:
            raise ValueError("max_ratio must be > 1.0")
        self.max_ratio = float(max_ratio)
        self.u_cap = min(float(u_cap), (self.max_ratio - 1.0) / self.max_ratio)  # §4.1 contract

        # Governor
        self.max_jump_up = float(max_jump_up)
        self.max_jump_down = float(max_jump_down)
        self.gov_noise_sensitivity = float(gov_noise_sensitivity)
        self._gov_jump_up_floor = 1.02
        self._gov_jump_down_floor = 0.75

        # Throttle
        # (v10.9) Dead trio k_stab/noise_db/stab_min REMOVED from signature.
        # Superseded by k_auth/auth_db/auth_min in v10.7.0; silent no-ops since.
        # Removal is deliberately loud (TypeError at stale call-sites) — the
        # silent-compat variant inverted the meaning of harness's k_stab=0.0
        # ("throttle off") without anyone noticing.
        self.boost_max = float(boost_max)
        self.noise_boost_thr = float(noise_boost_thr)
        self.boost_min_duration = int(boost_min_duration)
        self.headroom_margin = float(headroom_margin)
        self.wall_ema_fast_alpha = float(wall_ema_fast_alpha)
        self.wall_boost_thr = float(wall_boost_thr)
        # noise_ref
        self._noise_ref_alpha = _window_to_alpha(noise_ref_window)
        self.noise_ref_max_grow = float(noise_ref_max_grow)
        self.noise_ref_max_shrink = float(noise_ref_max_shrink)
        self.noise_ref_winsor_mult = float(noise_ref_winsor_mult)
        self.noise_ref_refractory_div = float(noise_ref_refractory_div)

        # Authority
        self.k_auth = float(k_auth)
        self.auth_db = float(auth_db)
        self.auth_min = float(auth_min)

        # Panic
        self.panic_symptom_thr = float(panic_symptom_thr)
        self.panic_confirm_steps = int(panic_confirm_steps)
        self.panic_factor_min = float(panic_factor_min)
        self.panic_decay_rate = float(panic_decay_rate)
        self.panic_recover_rate = float(panic_recover_rate)

        # N-adaptive panic threshold.
        # Binary channels → one player gives score ≈ 0.4 (full weight).
        # Fractional channels → one player gives score ≈ 0.65/N (diluted).
        # Scale threshold so "one player fully distressed" still qualifies
        # at small N, while requiring proportional distress at large N.
        # Formula: 2/(N+1) → N=2: ×0.667, N=4: ×0.400, N=8: ×0.222.
        # Floor 0.1 prevents trivial qualification at very large N.
        self._panic_symptom_thr_eff = max(
            self.panic_symptom_thr * 2.0 / (self.N + 1),
            0.1)

        # Tripwire trigger (sustained legacy calm)
        self.calm_legacy_confirm = int(calm_legacy_confirm)
        self.trip_len = int(trip_len)
        self.trip_cooldown = int(trip_cooldown)
        self.trip_auth_mult = float(trip_auth_mult)
        self.stress_alpha_event = float(stress_alpha_event)
        self.stress_alpha_tighten = float(stress_alpha_tighten)
        self.stress_alpha_relax = float(stress_alpha_relax)
        self.stress_auth_k = float(stress_auth_k)
        self.stress_auth_floor = float(stress_auth_floor)

        # Shock params
        self._shock_k = float(shock_k)
        self.snap_floor_mult = float(snap_floor_mult)
        self.shock_bypass_mult = float(shock_bypass_mult)
        self.shock_snap_floor_mult = float(shock_snap_floor_mult)
        self.shock_sustained_enable = bool(shock_sustained_enable)
        self.shock_hot_mult = float(shock_hot_mult)

        # Predictor
        self.osc_score_cap = float(osc_score_cap)
        self.osc_min_baseline = float(osc_min_baseline)
        self.osc_preempt_calm = bool(osc_preempt_calm)
        self.osc_require_high_vol = bool(osc_require_high_vol)
        self.osc_conf_thr = float(osc_conf_thr)

        # Hysteresis
        self.osc_risk_trigger_hi = float(osc_risk_trigger)
        _default_lo = float(osc_risk_trigger) * 0.75
        self.osc_risk_trigger_lo = float(
            osc_risk_trigger_lo if osc_risk_trigger_lo is not None else _default_lo)

        # Flip
        self.flip_win = int(flip_win)
        n_fp = max(flip_win, 2)
        self.flip_thr = max(0.1, 0.5 - 2.0 * (0.5 / math.sqrt(n_fp)))

        # Warmup
        self.warmup_steps = int(warmup_steps)

        # Convergence-based warmup
        self.convergence_cv_thr = float(convergence_cv_thr)
        self.convergence_growth_thr = float(convergence_growth_thr)
        self.convergence_n_confirm = int(convergence_n_confirm)
        self.max_warmup_steps = int(warmup_steps * max_warmup_mult)
        # (v10.9) Feasibility-aware convergence window.
        # Converged exit needs convergence_window × n_checks steps, where
        # n_checks = max(n_confirm+1, 3) + n_confirm - 1 (window boundaries
        # to fill min_windows, then n_confirm consecutive confirmations).
        # The old fixed default (200) made convergence ARITHMETICALLY
        # impossible for warmup_steps < ~400 → silent degraded mode with
        # deadband pinned at max (all v10.8.x CIFAR sweeps ran like that).
        _n_checks = (max(self.convergence_n_confirm + 1, 3)
                     + self.convergence_n_confirm - 1)
        if convergence_window is None:
            _w_auto = int(self.max_warmup_steps * 0.8 / max(_n_checks, 1))
            self.convergence_window = int(np.clip(_w_auto, 20, 200))
            self._convergence_window_auto = True
            if self.convergence_window * _n_checks > self.max_warmup_steps:
                warnings.warn(
                    f"GravBalancer: warmup_steps={warmup_steps} is too short for "
                    f"any convergence check (earliest converged exit "
                    f"{self.convergence_window * _n_checks} > max_warmup_steps "
                    f"{self.max_warmup_steps}). Warmup will exit DEGRADED "
                    f"(deadband pinned at dyn_deadband_z_max). Increase warmup_steps.")
        else:
            self.convergence_window = int(convergence_window)
            self._convergence_window_auto = False
            _earliest = self.convergence_window * _n_checks
            if _earliest > self.max_warmup_steps:
                raise ValueError(
                    f"Convergence arithmetically impossible: earliest converged exit "
                    f"= convergence_window({self.convergence_window}) × {_n_checks} "
                    f"= {_earliest} > max_warmup_steps({self.max_warmup_steps}). "
                    f"Warmup would ALWAYS exit degraded. Increase warmup_steps / "
                    f"max_warmup_mult, reduce convergence_window, or pass "
                    f"convergence_window=None for auto-scale.")
        self.ramp_i_gate = float(ramp_i_gate)
        # N-invariant: strict for small N, robust for large N
        if warmup_ok_frac_thr is not None:
            self._warmup_ok_frac_thr = float(warmup_ok_frac_thr)
        else:
            self._warmup_ok_frac_thr = 1.0 if self.N <= 2 else 0.8
        # Quantiles: strict (max) for N≤2, robust for N>2
        self._warmup_q = 1.0 if self.N <= 2 else float(warmup_q_robust)
        self._warmup_q_snap = 1.0 if self.N <= 2 else float(warmup_q_snap)
        self._ramp_steps_total = int(warmup_steps)  # ramp = min_warmup length
        # (v10.9) Immutable snapshots of constructor values mutated at runtime
        # (warmup calibration / degraded mode). reset_state() restores from
        # these — previously calibration leaked across resets: dyn_deadband_z
        # kept the prior run's calibrated value (3.5 after degraded) and
        # _ramp_steps_total doubled cumulatively on every degraded exit.
        self._init_dyn_deadband_z = float(dyn_deadband_z)
        self._init_ramp_steps_total = int(warmup_steps)

        # Climate control
        self._climate_alpha_fast_up = _window_to_alpha(climate_tau_fast)
        self._climate_alpha_fast_down = self._climate_alpha_fast_up * float(climate_relax_ratio_fast)
        self._climate_alpha_slow_up = _window_to_alpha(climate_tau_slow)
        self._climate_alpha_slow_down = self._climate_alpha_slow_up * float(climate_relax_ratio_slow)
        self.climate_k_meta = float(climate_k_meta)
        self.climate_lr_meta_floor = float(climate_lr_meta_floor)
        self.climate_E_hold_on = float(climate_E_hold_on)
        self.climate_E_hold_off = float(climate_E_hold_off)
        self.climate_slew_down = float(climate_slew_down)
        self.climate_slew_up = float(climate_slew_up)
        self._climate_E_deadband = float(climate_E_deadband)
        self._climate_E_winsor_mult = float(climate_E_winsor_mult)
        self._abs_lr_floor = float(abs_lr_floor) if abs_lr_floor is not None else self.base_lr * 0.001

        # Profile
        self.profile = profile
        if profile == "cooperative":
            if integrator_mode == "auto":
                self.integrator_mode = "on"
            self.boost_max = min(self.boost_max, 0.3)
        elif profile == "competitive":
            self.boost_max = min(self.boost_max, 0.1)

        # Debug
        self.debug_freeze = debug_freeze

        # --- Derived: calm ---
        _calm_raw = round(3.0 / max(self.d_filter_alpha, 1e-6))
        self.calm_len = int(np.clip(_calm_raw, 10, 200))
        _calm_target = 0.05
        _raw_cm = math.pow(_calm_target, 1.0 / max(self.calm_len, 1))
        self.calm_mult = float(np.clip(_raw_cm, 0.3, 0.95))

        # --- Derived: vol_thr adaptive ---
        self.vol_thr_k = 2.0
        self.vol_thr_clamp = (0.05, 3.0)
        self._rel_vol_ema = 0.0
        self._rel_vol_var_ema = 0.0
        self.vol_thr = 0.5

        # --- Derived: d_calm_thr adaptive ---
        self.d_calm_thr_k = 2.0
        self.d_calm_thr_clamp = (0.01, 5.0)
        self._d_mag_ema = 0.0
        self._d_mag_var_ema = 0.0
        self.d_calm_thr = 0.25

        _gap_noise_std = math.sqrt(self.stat_alpha / max(2.0 - self.stat_alpha, 1e-6))
        self._gap_noise_std = _gap_noise_std

        # =====================================================================
        # STATE
        # =====================================================================
        self.step = 0

        # Stats
        self.mus = np.zeros(self.N, dtype=np.float64)
        self.sigmas = np.ones(self.N, dtype=np.float64) * 1e-3
        self.ref_scale_ema = 1.0

        # Damper state
        self.u_p = np.zeros(self.N, dtype=np.float64)
        self.u_i = np.zeros(self.N, dtype=np.float64)
        self.prev_gaps = np.zeros(self.N, dtype=np.float64)
        self.d_filtered = np.zeros(self.N, dtype=np.float64)
        self._d_sigma = np.ones(self.N, dtype=np.float64) * 0.01  # running MAD of centered d
        self._d_baseline = np.zeros(self.N, dtype=np.float64)  # slow EMA of d_filtered
        self._d_base_alpha = _window_to_alpha(399)  # very slow baseline (~400 step window)

        # Integrator auto-gating state (§6)
        self._i_enabled = (self.integrator_mode != "off")
        self._sat_ema = 0.0
        self._hold_ema = 0.0
        self._flip_ema = 0.0
        self._gap_ema = 0.0
        self._gate_cooldown_remaining = 0
        self._i_toggle_count = 0
        self._gate_disable_reason_latched = 0  # set at disable, held until re-enable
        self._gate_disable_reason_now = 0      # current active triggers (live)

        # Flip detector
        self.prev_gap_sign = np.zeros(self.N, dtype=np.int8)
        self.flip_hist = deque(maxlen=self.flip_win)

        # Calm
        self.calm_ticks = 0
        self._calm_cause = 0

        # Comfort boost state
        self._quiet_ticks = 0
        self._wall_ema_fast = 0.0  # fast EMA tracking post-comfort wall hits
        self._prev_wall_final = False  # feedback: was comfort clipped last step?

        # noise_ref
        self._noise_ref = 0.0
        self._noise_ref_seeded = False
        self._noise_ref_update_reason = 0  # 0=normal, 1=winsor, 2=refractory, 3=rate-limit

        # Authority
        self._authority_factor = 1.0

        # Panic
        self._panic_factor = 1.0
        self._panic_symptom_counter = 0.0
        self._panic_active = False
        self._panic_symptom_score = 0.0
        self._panic_shock_stress = 0.0  # raw shock bypass signal
        self._panic_shock_stress_ema = 0.0  # filtered shock stress

        # Tripwire trigger counter
        self._calm_legacy_counter = 0

        # Tripwire state
        self._trip_active = False
        self._trip_ticks = 0
        self._trip_cooldown_ticks = 0

        # Control stress
        self._stress_wall_rate = 0.0
        self._stress_hold_rate = 0.0
        self._stress_usat_rate = 0.0
        self._control_stress = 0.0
        self._stress_auth_factor = 1.0

        # Intervention rate
        self._intervention_rate_ema = 0.0

        # Dynamic deadband
        self._dyn_deadband_z_eff = 0.0  # set from warmup; updated live
        self._dyn_deadband_z_eff_raw = 0.0  # pre-smoothing target

        # Output
        self.prev_lrs = np.full(self.N, self.base_lr, dtype=np.float64)
        self.last_metrics: dict = {}

        # Governor
        self._gov_jump_up = float(self.max_jump_up)
        self._gov_jump_down = float(self.max_jump_down)

        # Climate control
        self._climate_E = 1.0
        self._climate_E_fast = 1.0
        self._climate_E_slow = 1.0
        self._climate_lr_meta = 1.0
        self._climate_hold_active = False  # from prev step (1-step lag)
        self._prev_lrs_eff = None  # initialized at snaps_taken

        # Predictor
        self._pred_n = 3
        self._pred_fast = np.zeros(self._pred_n, dtype=np.float64)
        self._pred_slow = np.zeros(self._pred_n, dtype=np.float64)
        self._pred_seeded = False
        self._pred_scores = np.zeros(self._pred_n, dtype=np.float64)
        self.osc_risk = 0.0
        self.osc_conf = 0.0
        self._preempt_armed = True

        # Shock
        self._shock = 0.0
        self._shock_ema = 0.0
        self._shock_snap = 0.0
        self._shock_path = 0
        self._shock_thr = 0.0
        self._shock_scale = 0.0
        self._snap_floor = None
        self._warmup_snaps_taken = False
        self._snap_vol_thr = 0.0
        self._snap_d_calm_thr = 0.0

        # Convergence warmup state
        self._warmup_phase = 'A'  # 'A', 'B', 'ramp', 'active'
        self._warmup_converged = False
        self._ramp_factor = 0.0
        self._ramp_progress = 0.0
        self._ramp_steps_done = 0
        self._ramp_degraded = False  # True if B timed out
        self._degraded_reason = ''   # '' | 'timeout' | 'no_phase_b'
        # Per-player shock accumulator for convergence windows
        self._conv_shock_accum = np.zeros(self.N, dtype=np.float64)  # mus-change for convergence
        self._conv_resid_accum = np.zeros(self.N, dtype=np.float64)  # resid for snap (same units as _shock_ema)
        self._conv_window_count = 0  # steps in current window
        self._conv_window_means = []  # list of np.array(N,) — per-player mus-change window means
        self._conv_resid_window_means = []  # list of np.array(N,) — per-player resid window means
        self._conv_confirm_count = 0
        self._warmup_cv = 0.0  # diagnostic
        self._warmup_growth = 0.0  # diagnostic
        self._warmup_ok_frac = 0.0  # diagnostic
        self._conv_min_windows = max(self.convergence_n_confirm + 1, 3)

        # Transient per-step state (initialized here, set each step in adjust)
        self._prev_vals_warmup = None    # for per-player shock in warmup
        self._warmup_resid_pp = np.zeros(self.N, dtype=np.float64)
        self._auth_min_eff = self.auth_min

    # ================================================================= Reset (Step 5)

    def reset_state(self, force: bool = False) -> None:
        """Full state reset for clean re-run. Call before new experiment.
        Args:
            force: if False (default), raises if step > 0 (prevents mid-run reset).
        """
        if not force and self.step > 0:
            raise RuntimeError(
                f"reset_state() called at step={self.step}. "
                "Use reset_state(force=True) if intentional, or call before first adjust()."
            )
        self.step = 0
        self.mus = np.zeros(self.N, dtype=np.float64)
        self.sigmas = np.ones(self.N, dtype=np.float64) * 1e-3
        self.ref_scale_ema = 1.0
        self.u_p = np.zeros(self.N, dtype=np.float64)
        self.u_i = np.zeros(self.N, dtype=np.float64)
        self.prev_gaps = np.zeros(self.N, dtype=np.float64)
        self.d_filtered = np.zeros(self.N, dtype=np.float64)
        self._d_sigma = np.ones(self.N, dtype=np.float64) * 0.01
        self._d_baseline = np.zeros(self.N, dtype=np.float64)
        self._quiet_factor = 1.0
        self._warmup_z_samples = []
        # (v10.9) Restore from constructor snapshots. Previous behavior copied
        # the MUTATED runtime value into base — calibration (or the degraded
        # 3.5 pin) leaked into every subsequent run after reset.
        self.dyn_deadband_z = self._init_dyn_deadband_z
        self.dyn_deadband_z_base = self._init_dyn_deadband_z
        self._ramp_steps_total = self._init_ramp_steps_total
        self._warmup_z_n = 0
        self._warmup_z_pct_raw = float('nan')
        self._neg_proxy_warned = False
        self._i_enabled = (self.integrator_mode != "off")
        self._sat_ema = 0.0
        self._hold_ema = 0.0
        self._flip_ema = 0.0
        self._gap_ema = 0.0
        self._gate_cooldown_remaining = 0
        self._i_toggle_count = 0
        self._gate_disable_reason_latched = 0
        self._gate_disable_reason_now = 0
        self.prev_gap_sign = np.zeros(self.N, dtype=np.int8)
        self.flip_hist = deque(maxlen=self.flip_win)
        self.calm_ticks = 0
        self._calm_cause = 0
        self._quiet_ticks = 0
        self._wall_ema_fast = 0.0
        self._prev_wall_final = False

        # noise_ref
        self._noise_ref = 0.0
        self._noise_ref_seeded = False
        self._noise_ref_update_reason = 0

        # Authority
        self._authority_factor = 1.0

        # Panic
        self._panic_factor = 1.0
        self._panic_symptom_counter = 0.0
        self._panic_active = False
        self._panic_symptom_score = 0.0
        self._panic_shock_stress = 0.0  # raw shock bypass (bypasses deadband)
        self._panic_shock_stress_ema = 0.0  # filtered shock stress (asymmetric EMA)

        # Tripwire trigger counter
        self._calm_legacy_counter = 0

        # Tripwire state
        self._trip_active = False
        self._trip_ticks = 0
        self._trip_cooldown_ticks = 0

        # Control stress
        self._stress_wall_rate = 0.0
        self._stress_hold_rate = 0.0
        self._stress_usat_rate = 0.0
        self._control_stress = 0.0
        self._stress_auth_factor = 1.0

        # Intervention rate
        self._intervention_rate_ema = 0.0

        # Dynamic deadband
        self._dyn_deadband_z_eff = 0.0
        self._dyn_deadband_z_eff_raw = 0.0
        self.prev_lrs = np.full(self.N, self.base_lr, dtype=np.float64)
        self.last_metrics = {}
        self._gov_jump_up = float(self.max_jump_up)
        self._gov_jump_down = float(self.max_jump_down)

        # Climate control
        self._climate_E = 1.0
        self._climate_E_fast = 1.0
        self._climate_E_slow = 1.0
        self._climate_lr_meta = 1.0
        self._climate_hold_active = False
        self._prev_lrs_eff = None

        self._pred_fast = np.zeros(self._pred_n, dtype=np.float64)
        self._pred_slow = np.zeros(self._pred_n, dtype=np.float64)
        self._pred_seeded = False
        self._pred_scores = np.zeros(self._pred_n, dtype=np.float64)
        self.osc_risk = 0.0
        self.osc_conf = 0.0
        self._preempt_armed = True
        self._shock = 0.0
        self._shock_ema = 0.0
        self._shock_snap = 0.0
        self._shock_path = 0
        self._shock_thr = 0.0
        self._shock_scale = 0.0
        self._snap_floor = None
        self._warmup_snaps_taken = False
        self._snap_vol_thr = 0.0
        self._snap_d_calm_thr = 0.0

        # Convergence warmup
        self._warmup_phase = 'A'
        self._warmup_converged = False
        self._ramp_factor = 0.0
        self._ramp_progress = 0.0
        self._ramp_steps_done = 0
        self._ramp_degraded = False
        self._degraded_reason = ''
        self._conv_shock_accum = np.zeros(self.N, dtype=np.float64)
        self._conv_resid_accum = np.zeros(self.N, dtype=np.float64)
        self._conv_window_count = 0
        self._conv_window_means = []
        self._conv_resid_window_means = []
        self._conv_confirm_count = 0
        self._warmup_cv = 0.0
        self._warmup_growth = 0.0
        self._warmup_ok_frac = 0.0

        # Transient per-step state
        self._prev_vals_warmup = None
        self._warmup_resid_pp = np.zeros(self.N, dtype=np.float64)
        self._auth_min_eff = self.auth_min

        self._rel_vol_ema = 0.0
        self._rel_vol_var_ema = 0.0
        self.vol_thr = 0.5
        self._d_mag_ema = 0.0
        self._d_mag_var_ema = 0.0
        self.d_calm_thr = 0.25

    # ================================================================= API

    def adjust(self, *proxies) -> List[float]:
        vals = self._coerce_inputs(*proxies)
        self.step += 1
        eps = 1e-12

        # ━━━ 0. Shock (before stats update) ━━━
        if self.step >= 2:
            _resid_per_player = np.abs(vals - self.mus) / max(self.ref_scale_ema, eps)
            self._shock = float(np.mean(_resid_per_player))
            # Per-player resid for warmup snap calibration
            self._warmup_resid_pp = _resid_per_player.copy()
        else:
            self._shock = 0.0
            self._warmup_resid_pp = np.zeros(self.N, dtype=np.float64)
        # Maintain shock EMA (used by comfort gating)
        self._shock_ema = (1.0 - self.stat_alpha) * self._shock_ema + self.stat_alpha * self._shock

        # ━━━ 1. Update statistics ━━━
        self._update_stats(vals)

        # ━━━ 2. Compute gaps ━━━
        gaps, rel_vol, weights = self._compute_gaps()

        # === WARMUP (phases A/B) ===
        if self._warmup_phase in ('A', 'B'):
            return self._handle_warmup(gaps, rel_vol)

        # Snapshot hold state for THIS step (before _update_climate changes it)
        _climate_hold_applied = self._climate_hold_active

        # === RAMP progress (soft-start after warmup) ===
        if self._warmup_phase == 'ramp':
            self._ramp_steps_done += 1
            progress = min(1.0, self._ramp_steps_done / max(self._ramp_steps_total, 1))
            self._ramp_progress = progress  # linear progress [0,1]
            self._ramp_factor = progress * progress  # quadratic soft-start
            if self._ramp_steps_done >= self._ramp_steps_total:
                self._warmup_phase = 'active'
                self._ramp_factor = 1.0
                self._ramp_progress = 1.0

        # ━━━ 3. Steering (u_p + u_i) ━━━
        flip_ratio = self._update_flip_ratio(gaps)
        is_flipping = (flip_ratio > self.flip_thr)

        # D-filter (PT1)
        raw_d = gaps - self.prev_gaps
        self.prev_gaps = gaps.copy()
        self.d_filtered = (1.0 - self.d_filter_alpha) * self.d_filtered + self.d_filter_alpha * raw_d
        d_mag_mean = float(np.mean(np.abs(self.d_filtered)))

        # Update adaptive thresholds (freeze during calm to prevent self-reinforcing cycle:
        # calm → thresholds drift down → calm releases → normal noise exceeds lowered thr → calm again)
        if self.calm_ticks == 0 and not self._trip_active:
            self._update_adaptive_thresholds(rel_vol, d_mag_mean)

        # Predictor (slow baseline refractory during calm/shock —
        # same principle as adaptive thresholds: don't learn baseline on intervention-distorted data)
        raw_d_mean = float(np.mean(np.abs(raw_d)))
        integral_mean = float(np.mean(np.abs(self.u_i)))
        self._update_predictor(shock=self._shock, raw_d_mean=raw_d_mean,
                               integral_mean=integral_mean,
                               freeze_slow=(self.calm_ticks > 0 or self._trip_active))

        # §4.1 Damper — choose steering signal based on gaps_mode
        if self.gaps_mode == "dynamic":
            # Baseline-removed, self-normalized d_filtered
            self._d_baseline = ((1.0 - self._d_base_alpha) * self._d_baseline
                                + self._d_base_alpha * self.d_filtered)
            d_centered = self.d_filtered - self._d_baseline

            # Normalize by running MAD with separate floor (Step 4)
            d_abs = np.abs(d_centered)
            self._d_sigma = (1.0 - self.stat_alpha) * self._d_sigma + self.stat_alpha * d_abs
            d_floor = np.maximum(self._d_sigma, self.dyn_floor_min)
            gaps_steer = d_centered / d_floor

            # Deadband in z-score units (dynamic, scales with world drift)
            # world_drift = shock_ema / shock_snap: how much noisier is the world vs warmup
            # Smoothed via stress_alpha_event to prevent chatter at drift boundary
            if self._warmup_snaps_taken and self._shock_snap > 1e-12:
                world_drift = self._shock_ema / self._shock_snap
                raw_target = self.dyn_deadband_z_base * max(1.0, world_drift)
                raw_target = min(raw_target, self.dyn_deadband_z_max)
                self._dyn_deadband_z_eff_raw = raw_target
                # Asymmetric smoothing: widen fast, narrow slow
                # Uses existing stress_alpha_tighten/relax pair (same pattern as stress tracking)
                if raw_target > self._dyn_deadband_z_eff:
                    a_db = self.stress_alpha_tighten  # widen fast
                else:
                    a_db = self.stress_alpha_relax    # narrow slow
                self._dyn_deadband_z_eff = (1.0 - a_db) * self._dyn_deadband_z_eff + a_db * raw_target
            db_eff = self._dyn_deadband_z_eff if self._warmup_snaps_taken else self.dyn_deadband_z
            gaps_db = np.sign(gaps_steer) * np.maximum(np.abs(gaps_steer) - db_eff, 0.0)
        else:
            gaps_steer = gaps  # level-based
            gaps_db = gaps_steer.copy()
            gaps_db[np.abs(gaps_db) < self.gap_deadband] = 0.0

        targets = np.tanh(self.damper_k * gaps_db)

        # Anti-reverse: if target flips relative to current u_p and flipping detected
        if is_flipping:
            sign_conflict = (np.sign(targets) * np.sign(self.u_p)) < 0
            targets = np.where(sign_conflict, targets * self.reverse_damp, targets)

        self.u_p = (1.0 - self.beta_u) * self.u_p + self.beta_u * targets
        self.u_p = np.clip(self.u_p, -self.u_cap, self.u_cap)

        # (B) Quiet-gate: dampen commands when system is quiet
        z_max = float(np.max(np.abs(gaps_steer)))
        if self.quiet_gate_enable and self._warmup_snaps_taken and self.gaps_mode == "dynamic":
            if z_max > self.quiet_gate_bypass_z:
                self._quiet_factor = 1.0  # bypass: real event
            else:
                # smoothstep: rel_vol_ema in [snap_vol_thr * 0.5, snap_vol_thr] → [0, 1]
                lo = (self._noise_ref if self._noise_ref_seeded else self._snap_vol_thr) * 0.5
                hi = (self._noise_ref if self._noise_ref_seeded else self._snap_vol_thr)
                if self._rel_vol_ema <= lo:
                    self._quiet_factor = 0.0
                elif self._rel_vol_ema >= hi:
                    self._quiet_factor = 1.0
                else:
                    t = (self._rel_vol_ema - lo) / max(hi - lo, 1e-12)
                    self._quiet_factor = t * t * (3.0 - 2.0 * t)  # smoothstep
                # Floor: P never fully silenced
                self._quiet_factor = max(self._quiet_factor, self.quiet_gate_min)
        else:
            self._quiet_factor = 1.0

        # Apply quiet-gate to u_p (dampen, not clip)
        u_p_gated = self.u_p * self._quiet_factor

        # §4.2 Integrator
        calm_active, calm_cause, shock_trigger, shock_trip_trigger, mech_trigger, mech_risk_max = \
            self._evaluate_calm(gaps, rel_vol, is_flipping, d_mag_mean)

        # ━━━ Tripwire management ━━━
        if self._trip_active:
            self._trip_ticks -= 1
            if self._trip_ticks <= 0:
                self._trip_active = False
                self._trip_cooldown_ticks = self.trip_cooldown
        elif self._trip_cooldown_ticks > 0:
            self._trip_cooldown_ticks -= 1

        # Priority P1: calm disables integrator
        i_enabled_this_step = self._i_enabled and not calm_active and not self._trip_active
        # I-term gated during early ramp (prevent windup on uncalibrated commands)
        if self._warmup_phase == 'ramp':
            if self._ramp_degraded:
                i_enabled_this_step = False  # degraded: I off entire ramp
            elif self._ramp_progress < self.ramp_i_gate:
                i_enabled_this_step = False  # gate by linear progress, not quadratic ramp
        if self.debug_freeze == "no_integrator":
            i_enabled_this_step = False
        # Climate hold disables integrator (uses strong relax path)
        if _climate_hold_applied and self._warmup_phase == 'active':
            i_enabled_this_step = False

        u_total = u_p_gated + self.u_i

        # ━━━ 4. Calm / Shock / Mech corrections ━━━
        if calm_active:
            cc = calm_cause
            if cc == 1:  # SHOCK — hard stop (Step 3)
                u_total = np.zeros(self.N, dtype=np.float64)
                self.u_p[:] = 0.0
                self.u_i[:] = 0.0
            elif cc == 2:  # MECH
                u_total *= self.calm_mult
            else:  # LEGACY/PREEMPT
                u_total *= self.calm_mult

        if self.debug_freeze == "no_calm":
            calm_active = False
            u_total = u_p_gated + self.u_i

        # ━━━ 5. noise_ref + authority + panic (three-layer) ━━━
        self._update_noise_ref(calm_active or self._trip_active, shock_trigger or shock_trip_trigger)
        noise_norm = self._compute_noise_norm()
        authority_factor = self._compute_authority(noise_norm)
        # Multiply by stress factor and tripwire
        authority_factor *= self._stress_auth_factor
        if self._trip_active:
            authority_factor *= self.trip_auth_mult
        # Update intervention_rate BEFORE auth_min_eff so current step's
        # calm/trip state is reflected immediately (panic_active is prev-step, acceptable)
        _intervening = float(calm_active or self._trip_active or self._panic_active)
        _a_iv = self.stress_alpha_event
        self._intervention_rate_ema = ((1.0 - _a_iv) * self._intervention_rate_ema
                                       + _a_iv * _intervening)

        # Dynamic auth_min — idle when healthy, active when stressed/intervening
        # Uses existing stress_auth_k scale: auth_min_eff rises with health_proxy
        # health_proxy = max(stress, intervention_rate) — composite health signal
        # At health_proxy=0: auth_min_eff ≈ 0 (true idle)
        # At health_proxy >> halfpoint: auth_min_eff → auth_min (full base authority)
        # halfpoint derived from stress_auth: stress where stress_auth_factor = stress_auth_floor
        # = (1/floor - 1) / k. This is the stress level the system already considers "significant".
        _health_proxy = max(self._control_stress, self._intervention_rate_ema)
        _halfpoint = (1.0 / max(self.stress_auth_floor, 0.01) - 1.0) / max(self.stress_auth_k, 0.01)
        _auth_min_eff = self.auth_min * _health_proxy / (_health_proxy + _halfpoint + 1e-12)
        authority_factor = max(authority_factor, _auth_min_eff)
        self._authority_factor = authority_factor
        self._auth_min_eff = _auth_min_eff  # diagnostic
        panic_factor = self._panic_factor  # from previous step

        if self.debug_freeze == "no_throttle":
            authority_factor = 1.0
            panic_factor = 1.0

        # ━━━ 6. Ratio clamp (THE SINGLE CONTRACT WALL) ━━━
        u_total_zm_preclip = u_total - float(np.mean(u_total))  # before cap clip
        # Ramp_factor scales steering authority during soft-start
        _ramp = self._ramp_factor if self._warmup_phase in ('ramp',) else 1.0
        u_cap_eff = self.u_cap * authority_factor * _ramp
        # (v10.9) Plain clip after zero-mean broke the zero-sum invariant for
        # N>2 (asymmetric clipping shifts the mean → mean(lrs) drifted from
        # base_lr×panic_factor by up to ~10%). Project onto {mean=0} ∩ box.
        u_total_zm = self._project_zero_sum_box(u_total_zm_preclip, u_cap_eff)

        # Climate anti-chatter hold (1-step lag from prev step's E_fast)
        # Zero steering to prevent noise injection during storms.
        # I-term gated separately below. Hold only in active phase.
        if _climate_hold_applied and self._warmup_phase == 'active':
            u_total_zm = np.zeros(self.N, dtype=np.float64)
            u_total_zm_preclip = u_total_zm.copy()

        lrs_raw = self.base_lr * panic_factor * (1.0 + u_total_zm)
        lrs_clamped, lr_clamped_flag, clamp_frac = self._apply_ratio_clamp(lrs_raw)

        # ━━━ 7. Rate governor (first pass) ━━━
        lrs_gov, hold_lr = self._rate_governor(lrs_clamped)

        # ━━━ Saturation detection (Step 1: split steer vs wall) ━━━
        # (v10.9) Saturation is measured against the ACTIVE-phase allowance
        # (no ramp factor). During early ramp u_cap_eff ≈ 0, so any nonzero
        # command read as "saturated", polluting control_stress / panic /
        # climate-E with soft-start artifacts (E reached the hold threshold
        # 2.0 from saturation that existed only because the cap was ~1e-7).
        # In active phase _sat_cap == u_cap_eff — behavior unchanged.
        _sat_cap = max(self.u_cap * authority_factor, 1e-12)
        u_sat_steer = np.abs(u_total_zm) >= (_sat_cap - 1e-9)
        u_sat_steer_mean = float(np.mean(np.abs(u_total_zm) / _sat_cap))
        # Fraction of players at saturation (mean, N-invariant)
        u_sat_frac = float(np.mean(u_sat_steer.astype(float)))
        lr_wall_pre = lr_clamped_flag or bool(np.any(hold_lr))

        # ━━━ 8. Comfort throttle (≥1, after governor) ━━━
        # Update wall_ema_fast from PREVIOUS step's post-comfort wall hit
        wf_a = self.wall_ema_fast_alpha
        wf_val = 1.0 if self._prev_wall_final else 0.0
        self._wall_ema_fast = (1.0 - wf_a) * self._wall_ema_fast + wf_a * wf_val
        wall_rate_high = (self._wall_ema_fast > self.wall_boost_thr)

        comfort_factor_raw = self._compute_comfort(
            noise_norm=noise_norm,
            calm_active=calm_active,
            u_sat=u_sat_steer,
            lr_clamped_flag=lr_clamped_flag,
            hold_lr=hold_lr,
            wall_rate_high=wall_rate_high)

        if self.debug_freeze == "no_throttle":
            comfort_factor_raw = 1.0

        # Step 2: headroom-limit comfort so it never provokes final jump-clip
        floor = self.base_lr * 0.1
        low = np.maximum(self.prev_lrs * self._gov_jump_down, floor)
        high = self.prev_lrs * self._gov_jump_up
        headroom = high / np.maximum(lrs_gov, 1e-12)
        if not np.isfinite(headroom).all():
            c_max = 1.0
        else:
            c_max = max(float(np.min(headroom)) * (1.0 - self.headroom_margin), 1.0)  # margin
        comfort_limited_by_headroom = (comfort_factor_raw > c_max)
        comfort_factor = min(comfort_factor_raw, c_max)

        # Apply comfort + final jump-clip
        lrs_post_comfort = lrs_gov * comfort_factor
        lrs_final = np.clip(lrs_post_comfort, low, high)
        comfort_clipped = not np.allclose(lrs_post_comfort, lrs_final, rtol=1e-3, atol=1e-12)
        hold_lr_final = hold_lr | ~np.isclose(lrs_post_comfort, lrs_final, rtol=1e-3, atol=1e-12)
        lr_wall_final = lr_clamped_flag or bool(np.any(hold_lr_final))

        # Store for next-step feedback (quiet_ticks reset + wall_ema_fast)
        self._prev_wall_final = comfort_clipped or bool(np.any(hold_lr_final & ~hold_lr))

        _prev_lrs_snapshot = self.prev_lrs.copy()
        self.prev_lrs = lrs_final.copy()

        # Global wall signal — computed unconditionally for both integrator and diagnostics
        wall_lr_any = bool(np.any(hold_lr_final) or lr_clamped_flag)
        wall_any = bool(np.any(u_sat_steer) or wall_lr_any)  # for diagnostics (legacy)

        # ━━━ Integrator update (post-pipeline) ━━━
        if i_enabled_this_step:
            # Variant C anti-windup:
            # - LR walls (governor hold / ratio clamp) → global stop (hard)
            # - u_sat_steer → conditional: block only delta_i that pushes FURTHER into saturation
            # - Zero-sum projection reduces cross-coupling through zero-mean
            delta_i = self.ki * gaps_db
            delta_i = delta_i - float(np.mean(delta_i))  # zero-sum subspace

            if wall_lr_any:
                # Real LR wall hit → freeze all
                delta_i[:] = 0.0
            else:
                # Conditional: if steering saturated, block only "push further" components
                sat = np.abs(u_total_zm_preclip) >= (u_cap_eff - 1e-9)
                push_further = sat & (np.sign(u_total_zm_preclip) == np.sign(delta_i))
                delta_i[push_further] = 0.0

            self.u_i += delta_i
            self.u_i *= (1.0 - self.iterm_relax)
        else:
            # Stronger relax during ramp to prevent carrying warmup garbage
            _relax = self.iterm_relax_strong
            if self._warmup_phase == 'ramp':
                _relax = max(_relax, 0.3)  # aggressive drain during ramp
            self.u_i *= (1.0 - _relax)

        self.u_i = np.clip(self.u_i, -self.iterm_cap, self.iterm_cap)

        # ━━━ Auto-gating update (§6) ━━━
        self._update_auto_gating(u_sat_steer, hold_lr_final, lr_clamped_flag, flip_ratio,
                                 gaps_db, calm_active)

        # ━━━ Control stress update ━━━
        self._update_control_stress(wall_any, hold_lr_final, u_sat_steer_mean)

        # ━━━ Panic throttle update ━━━
        self._update_panic_throttle(
            wall_any=wall_any, flip_ratio=flip_ratio,
            hold_lr_final=hold_lr_final, lr_clamped_flag=lr_clamped_flag,
            calm_active=calm_active, calm_cause=calm_cause,
            u_sat_frac=u_sat_frac, clamp_frac=clamp_frac)

        # ━━━ Climate control update ━━━
        if self._warmup_snaps_taken:
            self._update_climate(
                resid_pp=self._warmup_resid_pp,
                shock_thr=self._shock_thr,
                wall_rate_ema=self._stress_wall_rate,
                u_sat_frac=u_sat_frac,
                clamp_frac=clamp_frac)

        # ━━━ Climate application (lr_meta + post-scale clip) ━━━
        if self._warmup_phase == 'active':
            _lr_meta_eff = self._climate_lr_meta
            lrs_eff = lrs_final * _lr_meta_eff

            if self._prev_lrs_eff is not None:
                _eff_low = np.maximum(self._prev_lrs_eff * self._gov_jump_down,
                                      self._abs_lr_floor)
                _eff_high = np.maximum(self._prev_lrs_eff * self._gov_jump_up,
                                       self._abs_lr_floor)
                lrs_eff = np.clip(lrs_eff, _eff_low, _eff_high)

            self._prev_lrs_eff = lrs_eff.copy()
        else:
            lrs_eff = lrs_final.copy()
            _lr_meta_eff = 1.0


        # ━━━ 9. Diagnostics (Step 0: extended sensors) ━━━
        mean_lr_final = float(np.mean(lrs_final))
        self.last_metrics = {
            "step": int(self.step),
            # Steering
            "u_p": self.u_p.copy(),
            "u_i": self.u_i.copy(),
            "u_total": u_total_zm.copy(),
            "u_sat_steer": u_sat_steer.astype(int),
            # LR walls (split: pre-comfort vs post-comfort)
            "lr_wall_pre": int(lr_wall_pre),
            "lr_wall_final": int(lr_wall_final),
            "hold_lr": hold_lr.astype(int),
            "hold_lr_final": hold_lr_final.astype(int),
            "lr_clamped_flag": int(lr_clamped_flag),
            "wall_lr_any": int(wall_lr_any),
            "wall_any": int(wall_any),
            # Comfort
            "comfort_factor_raw": float(comfort_factor_raw),
            "comfort_factor": float(comfort_factor),
            "comfort_active": int(comfort_factor > 1.0 + 1e-9),
            "comfort_clipped": int(comfort_clipped),
            "comfort_limited_by_headroom": int(comfort_limited_by_headroom),
            "headroom_min": float(c_max),  # after floor at 1.0
            "headroom_raw": float(np.min(headroom)) if np.isfinite(headroom).all() else 0.0,
            "wall_ema_fast": float(self._wall_ema_fast),
            "wall_rate_high": int(wall_rate_high),
            "prev_wall_final": int(self._prev_wall_final),
            # Integrator
            "i_enabled": int(i_enabled_this_step),
            "gate_disable_reason": int(self._gate_disable_reason_latched),
            "gate_disable_reason_now": int(self._gate_disable_reason_now),
            "sat_ema": float(self._sat_ema),
            "hold_ema": float(self._hold_ema),
            "flip_ema": float(self._flip_ema),
            "gap_ema": float(self._gap_ema),
            "toggle_count": int(self._i_toggle_count),
            # I-term activity diagnostics
            "i_sat": float(np.mean(np.abs(self.u_i)) / max(self.iterm_cap, 1e-12)),
            "gaps_db_nz": float(np.mean(np.abs(gaps_db) > 0)),
            # Throttle
            "authority_factor": float(authority_factor),
            "trip_active": int(self._trip_active),
            "control_stress": float(self._control_stress),
            "stress_auth_factor": float(self._stress_auth_factor),
            "panic_factor": float(self._panic_factor),
            "panic_active": int(self._panic_active),
            "panic_symptom_score": float(self._panic_symptom_score),
            "panic_shock_stress": float(self._panic_shock_stress),
            "panic_shock_stress_ema": float(self._panic_shock_stress_ema),
            "panic_symptom_thr_eff": float(self._panic_symptom_thr_eff),
            "noise_ref": float(self._noise_ref),
            "noise_ref_update_reason": int(self._noise_ref_update_reason),
            "u_cap_eff": float(u_cap_eff),
            "throttle": float(authority_factor * self._panic_factor * comfort_factor),
            "noise_norm": float(noise_norm),
            "auth_min_eff": float(self._auth_min_eff),
            "intervention_rate": float(self._intervention_rate_ema),
            "health_proxy": float(_health_proxy),
            "dyn_deadband_z_eff": float(self._dyn_deadband_z_eff),
            "dyn_deadband_z_eff_raw": float(self._dyn_deadband_z_eff_raw),
            "world_drift": float(self._shock_ema / max(self._shock_snap, 1e-12)) if self._warmup_snaps_taken else 0.0,
            "idle_fraction": float(np.mean(np.abs(u_total_zm) < 0.01)),
            "warmup_phase": self._warmup_phase,
            "warmup_converged": int(self._warmup_converged),
            "ramp_factor": float(self._ramp_factor),
            "ramp_degraded": int(self._ramp_degraded),
            "quiet_factor": float(self._quiet_factor),
            "dyn_deadband_z_base": float(self.dyn_deadband_z_base),
            "dyn_deadband_z": float(self.dyn_deadband_z),
            # Constraints — mean-anchored ratio (reflects actual clamp contract)
            "ratio_max_over_mean": float(np.max(lrs_final) / max(mean_lr_final, eps)),
            "ratio_mean_over_min": float(mean_lr_final / max(np.min(lrs_final), eps)),
            "ratio_raw": float(np.max(lrs_raw) / max(np.min(lrs_raw), eps)),
            "ratio_final": float(np.max(lrs_final) / max(np.min(lrs_final), eps)),
            # Calm
            "calm": int(calm_active),
            "calm_cause": int(calm_cause),
            "distress_level": (2 if (calm_cause in (1, 2) or self._panic_active) else
                              (1 if self._trip_active else 0)),
            "distress_kind": ("shock" if calm_cause == 1 else
                             "mech" if calm_cause == 2 else
                             "panic" if self._panic_active else
                             "shock_trip" if (self._trip_active and self._shock_path in (2,3)) else
                             "trip" if self._trip_active else "none"),
            "shock_trigger": int(shock_trigger),
            "shock_trip_trigger": int(shock_trip_trigger),
            "shock_path": int(self._shock_path),  # 0=NONE,1=BYPASS,2=SUSTAINED,3=NORMAL
            "mech_trigger": int(mech_trigger),
            "mech_risk_max": float(mech_risk_max),
            "shock": float(self._shock),
            "shock_ema": float(self._shock_ema),
            "shock_thr": float(self._shock_thr),
            "shock_scale": float(self._shock_scale),
            "shock_snap_diag": float(self._shock_snap),  # warmup snapshot (diagnostic only)
            # Gaps & vol
            "gaps": gaps.copy(),
            "gaps_steer": gaps_steer.copy(),
            "gaps_db": gaps_db.copy(),
            "z_max": float(np.max(np.abs(gaps_steer))),
            "rel_vol": float(rel_vol),
            "flip_ratio": float(flip_ratio),
            "osc_risk": float(self.osc_risk),
            "osc_conf": float(self.osc_conf),
            # Governor
            "gov_jump_up": float(self._gov_jump_up),
            "gov_jump_down": float(self._gov_jump_down),
            # LR pipeline stages
            "lrs_raw": lrs_raw.copy(),
            "lrs_gov": lrs_gov.copy(),
            "lrs_final": lrs_final.copy(),
            "prev_lrs": _prev_lrs_snapshot,
            # Climate control
            "climate_E": float(self._climate_E),
            "climate_E_fast": float(self._climate_E_fast),
            "climate_E_slow": float(self._climate_E_slow),
            "climate_lr_meta": float(self._climate_lr_meta),
            "climate_lr_meta_eff": float(_lr_meta_eff),
            "climate_hold_applied": int(_climate_hold_applied),
            "climate_hold_next": int(self._climate_hold_active),
            "climate_lr_eff_mean": float(np.mean(lrs_eff)),
            "lrs_eff": lrs_eff.copy(),
            "prev_lrs_eff": self._prev_lrs_eff.copy() if self._prev_lrs_eff is not None else np.full(self.N, self.base_lr),
        }

        return [float(x) for x in lrs_eff]

    # ================================================================= Warmup

    def _handle_warmup(self, gaps: np.ndarray, rel_vol: float) -> List[float]:
        """Handle warmup phases A and B.
        
        Dual-signal: accumulates both mus-change (for convergence check)
        and resid (for snap calibration, same units as _shock_ema).
        """
        if self.step == 1:
            warmup_d_mag = 0.0
        else:
            warmup_raw_d = gaps - self.prev_gaps
            warmup_d_mag = float(np.mean(np.abs(warmup_raw_d)))

        self.prev_lrs[:] = self.base_lr
        self.prev_gaps = gaps.copy()
        self.d_filtered[:] = 0.0
        self.u_p[:] = 0.0
        self.u_i[:] = 0.0
        s = np.sign(gaps).astype(np.int8)
        s[np.abs(gaps) < self.gap_deadband] = 0
        self.prev_gap_sign = s

        self._update_adaptive_thresholds(rel_vol, d_mag_mean=warmup_d_mag)
        # shock_ema already updated in adjust() before warmup branch

        # Update d_sigma EMA during warmup (needed for z-calibration)
        if self.step > 1:
            d_abs = np.abs(warmup_raw_d)
            self._d_sigma = (1.0 - self.stat_alpha) * self._d_sigma + self.stat_alpha * d_abs

        # Collect z samples for deadband calibration (phase A only)
        # Restrict z-collection to phase A. Extended warmup (phase B)
        # continued collecting z-samples with shrinking _d_sigma denominator,
        # inflating z-scores → deadband clipped to max (3.5) → paradoxical
        # "overprotection" degradation at low LR.
        if (self.dyn_deadband_z_auto
                and self._warmup_phase == 'A'
                and self.step > max(10, self.warmup_steps // 2)):
            d_floor = np.maximum(self._d_sigma, self.dyn_floor_min)
            if self.step > 1:
                z_approx = np.abs(warmup_raw_d) / d_floor
                self._warmup_z_samples.append(float(np.percentile(z_approx, 90)))

        # --- Per-player shock accumulation for convergence ---
        if self.step >= 2:
            eps = 1e-12
            shock_per_player = np.abs(self.mus - self._prev_vals_warmup) / max(self.ref_scale_ema, eps) if self._prev_vals_warmup is not None else np.zeros(self.N)
        else:
            shock_per_player = np.zeros(self.N)
        # Store current vals for next step's per-player shock
        self._prev_vals_warmup = self.mus.copy()  # mus already updated with current vals

        # Accumulate into convergence window
        # mus-change for convergence, resid for snap (dual-signal)
        self._conv_shock_accum += shock_per_player
        self._conv_resid_accum += self._warmup_resid_pp  # same units as _shock_ema
        self._conv_window_count += 1

        if self._conv_window_count >= self.convergence_window:
            # Window complete — push means
            wc = max(self._conv_window_count, 1)
            window_mean = self._conv_shock_accum / wc
            resid_mean = self._conv_resid_accum / wc
            self._conv_window_means.append(window_mean.copy())
            self._conv_resid_window_means.append(resid_mean.copy())
            self._conv_shock_accum[:] = 0.0
            self._conv_resid_accum[:] = 0.0
            self._conv_window_count = 0

        # --- Phase transitions ---
        if self._warmup_phase == 'A' and self.step >= self.warmup_steps:
            # Transition A → B
            self._warmup_phase = 'B'

        if self._warmup_phase == 'B':
            # Check convergence at window boundaries
            n_windows = len(self._conv_window_means)
            if n_windows >= self._conv_min_windows and self._conv_window_count == 0:
                converged = self._check_warmup_convergence()
                if converged:
                    self._warmup_converged = True
                    self._take_warmup_snaps(warmup_raw_d if self.step > 1 else None)
                    return self._finish_warmup_step(rel_vol)

            # Hard ceiling: max warmup reached → degraded exit
            if self.step >= self.max_warmup_steps:
                self._warmup_converged = False
                self._ramp_degraded = True
                self._degraded_reason = 'timeout'
                self._take_warmup_snaps(warmup_raw_d if self.step > 1 else None)
                return self._finish_warmup_step(rel_vol)

        # Phase A end-of-min-warmup: if phase B would have zero steps
        # (max_warmup_steps == warmup_steps), take snaps immediately
        if self._warmup_phase == 'B' and self.max_warmup_steps <= self.warmup_steps:
            self._warmup_converged = False
            self._ramp_degraded = True
            self._degraded_reason = 'no_phase_b'
            self._take_warmup_snaps(warmup_raw_d if self.step > 1 else None)
            return self._finish_warmup_step(rel_vol)

        return self._finish_warmup_step(rel_vol)

    def _finish_warmup_step(self, rel_vol: float) -> List[float]:
        """(v10.9) Single exit point for all warmup-phase returns.

        Previously the snap-taking paths returned early WITHOUT updating
        last_metrics, leaving the previous step's dict visible exactly on
        the step where warmup state changed.
        """
        self.last_metrics = {
            "warmup": True, "step": int(self.step),
            "warmup_phase": self._warmup_phase,
            "warmup_converged": int(self._warmup_converged),
            "ramp_factor": float(self._ramp_factor),
            "ramp_degraded": int(self._ramp_degraded),
            "degraded_reason": self._degraded_reason,
            "shock": float(self._shock),
            "rel_vol": float(rel_vol),
            "osc_risk": float(self.osc_risk),
            "osc_conf": float(self.osc_conf),
            "warmup_cv": float(self._warmup_cv),
            "warmup_growth": float(self._warmup_growth),
            "warmup_ok_frac": float(self._warmup_ok_frac),
            "conv_windows": len(self._conv_window_means),
            "conv_confirm": self._conv_confirm_count,
        }
        return [float(x) for x in self.prev_lrs]

    def _check_warmup_convergence(self) -> bool:
        """Check if shock_ema dynamics have stabilized.
        
        Per-player cv/growth with quantile aggregation for N-invariance.
        Returns True if converged.
        """
        nc = self.convergence_n_confirm
        n_needed = nc + 1  # need n_confirm+1 windows for growth calculation
        wm = self._conv_window_means
        
        if len(wm) < n_needed:
            return False
        
        # Use only last n_needed windows (not ancient history)
        recent = wm[-n_needed:]  # list of np.array(N,)
        # Stack into (n_needed, N) matrix
        M = np.stack(recent, axis=0)  # shape (n_needed, N)
        
        eps = 1e-12
        N = self.N
        
        # Per-player CV over recent windows
        cv_per_player = np.zeros(N)
        for i in range(N):
            col = M[:, i]
            mu = np.mean(col)
            cv_per_player[i] = float(np.std(col) / (mu + eps))
        
        # Per-player growth: geometric mean of log-ratios over consecutive windows
        growth_per_player = np.zeros(N)
        for i in range(N):
            col = M[:, i]
            log_ratios = []
            for j in range(1, len(col)):
                log_ratios.append(math.log((col[j] + eps) / (col[j-1] + eps)))
            mean_log_ratio = float(np.mean(log_ratios))
            growth_per_player[i] = math.exp(mean_log_ratio)
        
        # Per-player "ok" flag
        ok = ((cv_per_player < self.convergence_cv_thr) & 
              (growth_per_player < self.convergence_growth_thr))
        ok_frac = float(np.mean(ok))
        
        # Aggregate with quantile (N-invariant)
        q = self._warmup_q
        if N <= 2:
            cv_global = float(np.max(cv_per_player))
            growth_global = float(np.max(growth_per_player))
        else:
            cv_global = float(np.quantile(cv_per_player, q))
            growth_global = float(np.quantile(growth_per_player, q))
        
        # Store diagnostics
        self._warmup_cv = cv_global
        self._warmup_growth = growth_global
        self._warmup_ok_frac = ok_frac
        
        # Check criteria
        criteria_met = (cv_global < self.convergence_cv_thr and
                       growth_global < self.convergence_growth_thr and
                       ok_frac >= self._warmup_ok_frac_thr)
        
        if criteria_met:
            self._conv_confirm_count += 1
        else:
            self._conv_confirm_count = 0
        
        return self._conv_confirm_count >= self.convergence_n_confirm

    def _take_warmup_snaps(self, warmup_raw_d) -> None:
        """Take warmup snapshots and transition to ramp phase.
        
        Uses resid windows (|vals - mus| / ref_scale) for snap calibration,
        ensuring snap is in same units as runtime _shock_ema.
        Handles both normal (converged) and degraded exits.
        """
        scale = max(self.ref_scale_ema, 1e-12)
        floor_rel = max(self.min_volatility_rel, self.min_volatility / scale)
        snap_floor = max(self.snap_floor_mult * floor_rel, 1e-9)
        self._snap_floor = snap_floor

        # Snap from resid windows (same units as _shock_ema):
        # Convergence uses mus-change (smoother), but snap MUST use resid
        # because _shock_ema / _shock_thr / shock_hot / world_drift all live in resid space.
        resid_wm = self._conv_resid_window_means
        if len(resid_wm) > 0:
            recent_resid = resid_wm[-3:] if len(resid_wm) >= 3 else resid_wm
            per_player_snap = np.mean(np.stack(recent_resid), axis=0)  # shape (N,)
            if self.N > 2:
                snap_val = float(np.quantile(per_player_snap, self._warmup_q_snap))
            else:
                snap_val = float(np.max(per_player_snap))  # strict for N<=2
        else:
            snap_val = self._shock_ema  # scalar fallback (no windows yet)

        # Shock snap floor: don't let quiet warmup set unreasonably low threshold.
        shock_snap_vol_floor = self.shock_snap_floor_mult * self._rel_vol_ema
        self._shock_snap = max(snap_val, snap_floor, shock_snap_vol_floor)
        self._snap_vol_thr = float(np.clip(
            max(self.vol_thr, snap_floor),
            snap_floor,
            self.vol_thr_clamp[1]))  # ceiling: don't let noisy warmup inflate baseline
        self._snap_d_calm_thr = float(np.clip(
            max(self.d_calm_thr, snap_floor),
            snap_floor,
            self.d_calm_thr_clamp[1]))
        self._warmup_snaps_taken = True
        # Initialize effective LR tracking for climate post-scale clip.
        # base_lr is the correct anchor: during ramp, lrs_eff is not yet tracked,
        # and base_lr provides a stable reference for the first active step.
        self._prev_lrs_eff = np.full(self.N, self.base_lr, dtype=np.float64)
        # Seed noise_ref from warmup
        self._noise_ref = self._snap_vol_thr
        self._noise_ref_seeded = True

        # Warmup-calibrated deadband
        if self.dyn_deadband_z_auto:
            z_pct_raw = float('nan')
            if len(self._warmup_z_samples) >= 5:
                z_arr = np.array(self._warmup_z_samples)
                z_pct = float(np.percentile(z_arr, self.dyn_deadband_z_pct * 100))
                z_pct_raw = z_pct
                if z_pct > 2.0 * self.dyn_deadband_z_max:
                    self.dyn_deadband_z_base = float(np.clip(
                        self.dyn_deadband_z, self.dyn_deadband_z_min, self.dyn_deadband_z_max))
                else:
                    self.dyn_deadband_z_base = float(np.clip(
                        z_pct, self.dyn_deadband_z_min, self.dyn_deadband_z_max))
            else:
                self.dyn_deadband_z_base = float(np.clip(
                    self.dyn_deadband_z, self.dyn_deadband_z_min, self.dyn_deadband_z_max))
            self.dyn_deadband_z = self.dyn_deadband_z_base
        else:
            z_pct_raw = float('nan')
        self._warmup_z_n = len(self._warmup_z_samples)
        self._warmup_z_pct_raw = z_pct_raw
        self._warmup_z_samples = []  # free memory
        self._dyn_deadband_z_eff = self.dyn_deadband_z_base
        self._dyn_deadband_z_eff_raw = self.dyn_deadband_z_base

        # Seed dynamic gaps baseline
        self._d_baseline[:] = 0.0
        if warmup_raw_d is not None:
            warmup_d_abs = np.abs(warmup_raw_d)
            self._d_sigma = np.maximum(self._d_sigma, np.maximum(warmup_d_abs, self.dyn_floor_min))
        else:
            self._d_sigma = np.maximum(self._d_sigma, self.dyn_floor_min)

        # Degraded mode adjustments
        if self._ramp_degraded:
            # (v10.9) Degraded exit is no longer silent — it pins the deadband
            # at max and disables I-term for the whole ramp, i.e. the steering
            # is nearly inert for the rest of the run.
            warnings.warn(
                f"GravBalancer warmup exited DEGRADED at step {self.step} "
                f"(reason: {self._degraded_reason or 'timeout'}): convergence not "
                f"confirmed within max_warmup_steps={self.max_warmup_steps}. "
                f"Deadband pinned at {self.dyn_deadband_z_max}σ; steering will be "
                f"nearly inert. Check warmup_steps vs convergence_window.")
            self._ramp_steps_total = self._ramp_steps_total * 2
            self.dyn_deadband_z_base = self.dyn_deadband_z_max
            self.dyn_deadband_z = self.dyn_deadband_z_max
            self._dyn_deadband_z_eff = self.dyn_deadband_z_max
            self._dyn_deadband_z_eff_raw = self.dyn_deadband_z_max

        # Transition to ramp phase
        self._warmup_phase = 'ramp'
        self._ramp_steps_done = 0
        self._ramp_factor = 0.0
        # Clean up convergence tracking memory
        self._conv_window_means = []
        self._conv_resid_window_means = []
        self._conv_shock_accum = np.zeros(self.N, dtype=np.float64)
        self._conv_resid_accum = np.zeros(self.N, dtype=np.float64)
        # Clean up warmup vals tracker
        self._prev_vals_warmup = None

    # ================================================================= Stats

    def _coerce_inputs(self, *proxies) -> np.ndarray:
        if len(proxies) == 1 and isinstance(proxies[0], (list, tuple, np.ndarray)):
            proxies = tuple(proxies[0])
        if len(proxies) != self.N:
            raise ValueError(f"Expected {self.N} proxies, got {len(proxies)}")
        arr = np.asarray(proxies, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("Proxies contain NaN/Inf.")
        neg = arr < 0.0
        if np.any(neg):
            # (v10.9, amendment П1) Contract: proxy >= 0, with a tolerance
            # band. Tiny negatives (FP noise in custom proxies) are clamped
            # with a ONE-TIME warning — a hard raise here would kill a long
            # run over -1e-12 at step 50000 (anti-survival). Gross negatives
            # still raise: they mean the proxy is sign-changing and half its
            # dynamic range would be silently destroyed (the v10.8 failure
            # mode this contract exists to prevent).
            tol = 1e-6 * float(np.mean(np.abs(arr))) + 1e-12
            if np.any(arr < -tol):
                raise ValueError(
                    f"GravBalancer proxies must be non-negative, got {arr.tolist()}. "
                    "Internal scales (ref_scale_ema, rel_vol, shock) assume a positive "
                    "scale. For sign-changing losses wrap them monotonically, e.g. "
                    "softplus(loss), instead of passing raw or ReLU-clipped values.")
            if not self._neg_proxy_warned:
                warnings.warn(
                    f"GravBalancer: tiny negative proxy clamped to 0 "
                    f"(min={float(arr.min()):.3e}, tol={tol:.3e}). "
                    "Warned once; further occurrences are clamped silently.")
                self._neg_proxy_warned = True
            arr = np.where(neg, 0.0, arr)
        return arr

    def _update_stats(self, vals: np.ndarray) -> None:
        a = self.stat_alpha
        curr_scale = float(np.mean(vals) + 1e-12)

        if self.step == 1:
            self.mus = vals.copy()
            self.sigmas[:] = 1e-3
            self.ref_scale_ema = curr_scale
            return

        self.ref_scale_ema = (1.0 - a) * self.ref_scale_ema + a * curr_scale
        self.mus = (1.0 - a) * self.mus + a * vals
        diffs = np.abs(vals - self.mus)
        self.sigmas = (1.0 - a) * self.sigmas + a * diffs

    # ================================================================= Gaps (§3)

    def _compute_gaps(self) -> Tuple[np.ndarray, float, np.ndarray]:
        eps = 1e-12
        scale = float(max(self.ref_scale_ema, eps))
        vol_floor = np.maximum(self.sigmas,
                               np.maximum(self.min_volatility, self.min_volatility_rel * scale))
        vol_floor = np.maximum(vol_floor, eps)

        sigma_soft = np.maximum(self.sigmas, 1e-9)
        weights = 1.0 / sigma_soft
        weights = weights / np.sum(weights)

        consensus = float(np.sum(weights * self.mus))
        gaps = (self.mus - consensus) / vol_floor
        rel_vol = float(np.mean(self.sigmas) / scale)
        return gaps, rel_vol, weights

    # ================================================================= Calm (§7)

    def _evaluate_calm(self, gaps: np.ndarray, rel_vol: float,
                       is_flipping: bool, d_mag_mean: float
                       ) -> Tuple[bool, int, bool, bool, bool, float]:
        eps = 1e-12

        # Mechanical risk
        correction = np.abs(self.u_p)
        braking_proxy = np.abs(self.d_filtered)
        ctrl_ratio = correction / (braking_proxy + eps)
        approach = (gaps * self.d_filtered) < 0
        gap_abs = np.abs(gaps)
        active = gap_abs > (3.0 * self.gap_deadband)
        speed_ratio = np.where(
            active,
            np.abs(self.d_filtered) / (gap_abs + self.gap_deadband),
            0.0)
        speed_ratio = np.clip(speed_ratio, 0.0, 5.0)
        risk_raw = speed_ratio * np.minimum(ctrl_ratio, 3.0) / 3.0
        mech_risk = np.where(approach & active, np.clip(risk_raw, 0.0, 1.0), 0.0)
        mech_risk_max = float(np.max(mech_risk))

        # Shock trigger (operational threshold, not warmup snapshot)
        # - shock_scale = max(noise_ref, snap_vol_thr, eps) — live operational norm
        # - BYPASS → calm (true emergency), SUSTAINED/NORMAL → tripwire (not calm)
        # shock_path: 0=NONE, 1=BYPASS, 2=SUSTAINED, 3=NORMAL
        shock_trigger = False
        shock_trip_trigger = False  # SUSTAINED/NORMAL → trip, not calm
        self._shock_path = 0
        self._shock_thr = 0.0
        self._shock_scale = 0.0
        if self._warmup_snaps_taken:
            # Operational threshold (live noise_ref, not frozen warmup snapshot)
            shock_scale = max(
                (self._noise_ref if self._noise_ref_seeded else self._snap_vol_thr),
                self._snap_vol_thr, 1e-12)
            thr = self._shock_k * shock_scale
            self._shock_thr = thr  # store for diagnostics
            self._shock_scale = shock_scale

            if self._shock > self.shock_bypass_mult * thr:
                shock_trigger = True       # → calm (cause=1)
                self._shock_path = 1       # BYPASS: real emergency
            elif self.shock_sustained_enable and self._shock_ema > thr:
                shock_trip_trigger = True   # → tripwire, NOT calm
                self._shock_path = 2       # SUSTAINED
            elif self._rel_vol_ema >= self._snap_vol_thr:
                if self._shock > thr:
                    shock_trip_trigger = True  # → tripwire, NOT calm
                    self._shock_path = 3       # NORMAL

        mech_trigger = (mech_risk_max > 0.5)

        # Calm state machine
        calm_cause = self._calm_cause
        if self.calm_ticks == 0:
            if shock_trigger:
                # Only BYPASS (path=1) → real emergency calm
                self.calm_ticks = self.calm_len
                self._calm_cause = 1
                calm_cause = 1
            elif shock_trip_trigger:
                # SUSTAINED/NORMAL shock → tripwire, not calm
                if (self._trip_cooldown_ticks == 0 and not self._trip_active):
                    self._trip_active = True
                    self._trip_ticks = self.trip_len
            elif mech_trigger:
                self.calm_ticks = self.calm_len
                self._calm_cause = 2 if (self._rel_vol_ema < self._gap_noise_std) else 3
                calm_cause = self._calm_cause
            else:
                legacy_trigger = (rel_vol > self.vol_thr) and (
                    is_flipping or d_mag_mean > self.d_calm_thr)

                if self._preempt_armed:
                    preempt_ok = (not self.osc_require_high_vol) or (rel_vol > self.vol_thr)
                    preempt_trigger = (
                        self.osc_preempt_calm and preempt_ok
                        and (self.osc_risk >= self.osc_risk_trigger_hi)
                        and (self.osc_conf >= self.osc_conf_thr))
                    if preempt_trigger:
                        self.calm_ticks = self.calm_len
                        self._calm_cause = 4
                        calm_cause = 4
                        self._preempt_armed = False
                else:
                    if self.osc_risk <= self.osc_risk_trigger_lo:
                        self._preempt_armed = True

                # Sustained confirmation → tripwire (not full calm)
                if legacy_trigger:
                    self._calm_legacy_counter += 1
                else:
                    self._calm_legacy_counter = 0

                # Tripwire: short intervention, not emergency calm
                if (self._calm_legacy_counter >= self.calm_legacy_confirm
                        and self.calm_ticks == 0
                        and self._trip_cooldown_ticks == 0
                        and not self._trip_active):
                    self._trip_active = True
                    self._trip_ticks = self.trip_len
                    self._calm_legacy_counter = 0
                    # NOTE: we do NOT set calm_ticks or calm_cause=3 here

        calm_active = (self.calm_ticks > 0)
        if calm_active:
            self.calm_ticks -= 1
        else:
            self._calm_cause = 0
            calm_cause = 0

        return calm_active, calm_cause, shock_trigger, shock_trip_trigger, mech_trigger, mech_risk_max

    # ================================================================= Throttle (§8)

    def _compute_noise_norm(self) -> float:
        eps = 1e-12
        if self._noise_ref_seeded:
            ref = max(self._noise_ref, eps)
        elif self._warmup_snaps_taken:
            ref = max(self._snap_vol_thr, eps)
        else:
            ref = max(self.vol_thr, eps)
        return self._rel_vol_ema / ref

    def _compute_authority(self, noise_norm: float) -> float:
        """Layer 1: authority factor. Scales u_cap, not base_lr.
        Returns raw authority WITHOUT floor — floor is applied once
        in adjust() via dynamic auth_min_eff."""
        x = max(0.0, noise_norm - self.auth_db)
        authority = 1.0 / (1.0 + self.k_auth * x)
        return authority  # no floor here; dynamic floor in adjust()

    def _compute_comfort(self, noise_norm: float, calm_active: bool,
                         u_sat: np.ndarray, lr_clamped_flag: bool,
                         hold_lr: np.ndarray, wall_rate_high: bool) -> float:
        """Comfort factor (§8.2). Gated by: calm, u_sat, lr_clamped, hold_lr, shock_ema, wall_rate."""
        # All gates must be clear
        u_sat_any = np.any(u_sat)
        hold_any = np.any(hold_lr)
        shock_hot = (self._shock_ema > self._shock_snap * self.shock_hot_mult) if self._warmup_snaps_taken else False

        # Also gate on wall_rate_high (post-comfort wall feedback)
        if calm_active or u_sat_any or lr_clamped_flag or hold_any or shock_hot or wall_rate_high or self.boost_max <= 0:
            self._quiet_ticks = 0
            return 1.0

        # Feedback: if LAST step hit a post-comfort wall, reset quiet_ticks
        if self._prev_wall_final:
            self._quiet_ticks = 0

        if noise_norm < self.noise_boost_thr:
            self._quiet_ticks += 1
        else:
            self._quiet_ticks = 0

        if self._quiet_ticks < self.boost_min_duration:
            return 1.0

        z = max(0.0, self.noise_boost_thr - noise_norm) / max(self.noise_boost_thr, 1e-9)
        return 1.0 + self.boost_max * z

    # ================================================================= Zero-sum projection (§9)

    def _project_zero_sum_box(self, u: np.ndarray, cap: float) -> np.ndarray:
        """(v10.9) Project a vector onto {mean = 0} ∩ [-cap, cap]^N.

        Alternating projection (clip ↔ re-center); the intersection is
        non-empty (0 lies in both sets), convergence is fast in practice.
        For N=2 the zero-mean vector is symmetric (±a), so plain clip was
        already correct — behavior unchanged. For N>2 this restores the
        documented invariant mean(lrs_raw) = base_lr × panic_factor.
        """
        if cap <= 0.0:
            return np.zeros_like(u)
        v = u - float(np.mean(u))
        for _ in range(50):
            v_c = np.clip(v, -cap, cap)
            drift = float(np.mean(v_c))
            if abs(drift) <= 1e-15:
                return v_c
            v = v_c - drift
        return np.clip(v, -cap, cap)

    # ================================================================= Ratio clamp (§9)

    def _apply_ratio_clamp(self, lrs: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        mean_lr = float(np.mean(lrs))
        lower = mean_lr / self.max_ratio
        upper = mean_lr * self.max_ratio
        clamped = np.clip(lrs, lower, upper)

        # Preserve mean
        mean_after = float(np.mean(clamped) + 1e-12)
        corrected = clamped * (mean_lr / mean_after)
        corrected = np.clip(corrected, lower, upper)

        flag = not np.allclose(lrs, corrected, rtol=1e-3, atol=1e-12)
        # Fraction of players clamped (N-invariant)
        clamp_frac = float(np.mean(~np.isclose(lrs, corrected, rtol=1e-3, atol=1e-12)))
        return corrected, flag, clamp_frac


    # ================================================================= noise_ref

    def _update_noise_ref(self, calm_active: bool, shock_trigger: bool) -> None:
        """Update live operational noise norm. Robustly: winsorize + rate-limit + refractory."""
        if not self._noise_ref_seeded:
            return

        raw = self._rel_vol_ema
        ref = self._noise_ref
        alpha = self._noise_ref_alpha
        reason = 0  # 0=normal

        # Winsorize: clip input relative to current noise_ref
        max_input = ref * self.noise_ref_winsor_mult
        if raw > max_input:
            raw = max_input
            reason = 1  # winsorized

        # Refractory: during shock/calm, learn much slower
        if calm_active or shock_trigger:
            alpha = alpha / self.noise_ref_refractory_div
            reason = max(reason, 2)  # refractory

        # EMA update
        new_ref = (1.0 - alpha) * ref + alpha * raw

        # Rate-limit
        if new_ref > ref * self.noise_ref_max_grow:
            new_ref = ref * self.noise_ref_max_grow
            reason = max(reason, 3)  # rate-limited
        elif new_ref < ref * self.noise_ref_max_shrink:
            new_ref = ref * self.noise_ref_max_shrink
            reason = max(reason, 3)

        # Sanity floor
        new_ref = max(new_ref, 1e-12)

        self._noise_ref = new_ref
        self._noise_ref_update_reason = reason

    # ================================================================= Control stress

    def _update_control_stress(self, wall_any: bool, hold_lr: "np.ndarray",
                                u_sat_mean: float) -> None:
        """Update control stress: EMA of event rates → asymmetric authority squeeze.

        Pipeline: raw events → event_rate_ema → composite_stress → fast-tighten/slow-relax → auth_factor
        """
        a_ev = self.stress_alpha_event

        # Stage 1: smooth event rates (don't let single spikes jerk authority)
        self._stress_wall_rate = (1 - a_ev) * self._stress_wall_rate + a_ev * float(wall_any)
        self._stress_hold_rate = (1 - a_ev) * self._stress_hold_rate + a_ev * float(np.any(hold_lr))
        self._stress_usat_rate = (1 - a_ev) * self._stress_usat_rate + a_ev * u_sat_mean

        # Stage 2: composite stress (0..1)
        # wall and hold are strong signals; u_sat is frequent early signal
        raw_stress = (0.35 * self._stress_wall_rate
                    + 0.25 * self._stress_hold_rate
                    + 0.40 * self._stress_usat_rate)
        raw_stress = min(raw_stress, 1.0)

        # Stage 3: asymmetric tracking (fast tighten, slow relax)
        if raw_stress > self._control_stress:
            alpha = self.stress_alpha_tighten
        else:
            alpha = self.stress_alpha_relax
        self._control_stress = (1 - alpha) * self._control_stress + alpha * raw_stress

        # Stage 4: stress → authority factor (1/(1 + k*stress), floored)
        self._stress_auth_factor = max(
            1.0 / (1.0 + self.stress_auth_k * self._control_stress),
            self.stress_auth_floor)

    # ================================================================= Panic throttle

    def _update_panic_throttle(self, wall_any: bool, flip_ratio: float,
                                hold_lr_final: np.ndarray, lr_clamped_flag: bool,
                                calm_active: bool, calm_cause: int,
                                u_sat_frac: float = 0.0,
                                clamp_frac: float = 0.0) -> None:
        """Layer 2: rare preemptive brake on base_lr.
        Triggered by compound loss-of-control symptoms, not raw noise.

        Channels are N-invariant: fractional (mean-of-players), not binary.
        N-adaptive threshold: thr_eff = base × 2/(N+1), floor 0.1.
        shock_stress_ema filters transient spikes. ±1 counter with forgetting."""

        # ── Compound symptom score (normalized ~[0, 1]) ──
        #
        # N-INVARIANT channels.
        # Old: wall_score = 1.0 if ANY player hits wall → P(fire) = 1-(1-p)^N
        # New: wall_score = max(u_sat_frac, hold_frac) where u_sat_frac is
        #      fraction of players AT saturation (boolean mean, not continuous
        #      ratio). This scales with fraction of players in trouble.
        # Same for clamp: fraction of players clamped, not boolean.
        # IMPORTANT: u_sat_frac = mean(u_sat_steer), NOT u_sat_steer_mean
        #   (which is mean(|u|/u_cap) — continuous, always > 0, wrong signal).
        hold_score = float(np.mean(hold_lr_final.astype(float))) if hold_lr_final.size > 0 else 0.0
        wall_score = max(u_sat_frac, hold_score)  # ∈ [0,1], N-invariant
        flip_score = min(flip_ratio / max(self.flip_thr, 0.01), 1.0)  # 0..1
        clamp_score = clamp_frac                         # ∈ [0,1], N-invariant

        # Raw shock distress — bypasses deadband completely.
        if self._warmup_snaps_taken and self._shock_thr > 1e-12:
            shock_ratio = self._shock_ema / self._shock_thr
            shock_stress = min(max(0.0, shock_ratio - 1.0), 1.0)
        else:
            shock_stress = 0.0
        self._panic_shock_stress = shock_stress

        # Asymmetric EMA of shock_stress.
        if shock_stress > self._panic_shock_stress_ema:
            _a_ss = self.stress_alpha_tighten   # rising: track quickly
        else:
            _a_ss = self.stress_alpha_relax     # falling: release slowly
        self._panic_shock_stress_ema = (
            (1.0 - _a_ss) * self._panic_shock_stress_ema + _a_ss * shock_stress)

        # Strongest distress channel: wall fraction vs persistent shock.
        wall_or_shock = max(wall_score, self._panic_shock_stress_ema)
        score = 0.4 * wall_or_shock + 0.2 * flip_score + 0.25 * hold_score + 0.15 * clamp_score
        self._panic_symptom_score = score

        # ── Counter reset: only non-shock calm ──
        if calm_active and calm_cause != 1:
            self._panic_symptom_counter = 0.0
            if self._panic_active:
                self._panic_factor = min(self._panic_factor + self.panic_recover_rate, 1.0)
                if self._panic_factor >= 1.0 - 1e-6:
                    self._panic_active = False
            return

        # ── Counter update: ±1 with forgetting policy ──
        #
        # Gate: physical evidence of distress (wall fraction or shock breach).
        has_distress_evidence = (wall_or_shock > 1e-6)

        # Simple ±1 with N-invariant symptom channels + forgetting policy.
        # With fractional wall_score/clamp_score, threshold crossings mean
        # "systemic distress", not "one player sneezed". Rate modulation
        # Rate modulation was compensating for inflated binary channels — now
        # unnecessary. EMA on shock_stress handles transient filtering.
        #
        # Forgetting policy:
        #   distress + qualifying  → counter += 1  (accumulate)
        #   distress + !qualifying → counter holds  (fire burning, don't forget)
        #   !distress              → counter -= 1  (fire out, forget fast)
        #
        # Requires u_sat_frac (boolean fraction), NOT u_sat_steer_mean
        # (continuous ratio always > 0 — would make distress always true).
        if has_distress_evidence and score >= self._panic_symptom_thr_eff - 1e-6:
            self._panic_symptom_counter += 1
        elif not has_distress_evidence:
            self._panic_symptom_counter = max(0.0, self._panic_symptom_counter - 1)
        # else: distress but sub-threshold score → hold (don't forget mid-fire)

        if not self._panic_active:
            if self._panic_symptom_counter >= self.panic_confirm_steps:
                self._panic_active = True
                self._panic_symptom_counter = 0.0

        if self._panic_active:
            if score >= self._panic_symptom_thr_eff * 0.5:
                self._panic_factor = max(
                    self._panic_factor - self.panic_decay_rate,
                    self.panic_factor_min)
            else:
                self._panic_factor = min(
                    self._panic_factor + self.panic_recover_rate, 1.0)
                if self._panic_factor >= 1.0 - 1e-6:
                    self._panic_active = False
                    self._panic_factor = 1.0
        else:
            self._panic_factor = 1.0

    # ================================================================= Climate control

    def _update_climate(self, resid_pp: np.ndarray, shock_thr: float,
                        wall_rate_ema: float, u_sat_frac: float,
                        clamp_frac: float) -> None:
        """Energy-based climate control: E → E_fast/E_slow → lr_meta + hold.

        Computed at end of each step from observable signals.
        Hold state affects NEXT step (1-step lag, breaks algebraic loop).
        lr_meta applied at end-of-pipeline to produce lrs_eff.

        E channels (all ≥0, E=1 ↔ calm):
          - shock_ratio_hi: per-player resid / shock_thr, N-invariant aggregation
          - wall_rate_ema: EMA of wall event rate (from control_stress)
          - u_sat_frac: fraction of players at steering saturation
          - clamp_frac: fraction of players at ratio clamp
        """
        eps = 1e-12

        # ── E: single scalar storm metric ──
        if shock_thr > eps and resid_pp.size > 0:
            _ratios = resid_pp / (shock_thr + eps)
            if self.N <= 2:
                shock_ratio_hi = float(np.max(_ratios))
            else:
                shock_ratio_hi = float(np.quantile(_ratios, 0.9))
        else:
            shock_ratio_hi = 0.0

        E = max(shock_ratio_hi,
                1.0 + wall_rate_ema,
                1.0 + u_sat_frac,
                1.0 + clamp_frac)
        self._climate_E = E  # diagnostics keep the RAW value

        # ── E_fast: catches spikes (tau ~30 steps) — RAW E by design ──
        # (anti-chatter hold is the spike reflex; winsorizing it would dull
        # exactly the reaction it exists for)
        if E > self._climate_E_fast:
            a_fast = self._climate_alpha_fast_up      # tighten: track quickly
        else:
            a_fast = self._climate_alpha_fast_down     # relax: release slowly
        self._climate_E_fast = (1.0 - a_fast) * self._climate_E_fast + a_fast * E

        # ── E_slow: chronic storm (tau ~1000 steps) — WINSORIZED input ──
        # (v10.9.1, К1) Clip E relative to current E_slow (pattern: noise_ref
        # winsorization). Unwinsorized single spikes (E≈8021 observed on toy
        # 1e-2, 2026-06-10) ratcheted E_slow into ~46 epochs of braking —
        # "chronic storm tracker" degenerated into scar tissue from one event.
        # A sustained storm still raises E_slow exponentially (up to ×winsor
        # per crossing); a single spike contributes at most one bounded step.
        if self._climate_E_winsor_mult > 0:
            E_sl = min(E, max(self._climate_E_slow, 1.0) * self._climate_E_winsor_mult)
        else:
            E_sl = E
        if E_sl > self._climate_E_slow:
            a_slow = self._climate_alpha_slow_up       # tighten
        else:
            a_slow = self._climate_alpha_slow_down     # relax (very slow)
        self._climate_E_slow = (1.0 - a_slow) * self._climate_E_slow + a_slow * E_sl

        # ── lr_meta = f(E_slow): climate control signal ──
        x = max(0.0, self._climate_E_slow - 1.0 - self._climate_E_deadband)
        lr_meta_target = 1.0 / (1.0 + self.climate_k_meta * x)
        lr_meta_target = max(lr_meta_target, self.climate_lr_meta_floor)
        lr_meta_target = min(lr_meta_target, 1.0)

        # Slew-rate limited transition (down fast, up slow)
        if lr_meta_target < self._climate_lr_meta:
            self._climate_lr_meta = max(
                self._climate_lr_meta - self.climate_slew_down,
                lr_meta_target)
        else:
            self._climate_lr_meta = min(
                self._climate_lr_meta + self.climate_slew_up,
                lr_meta_target)

        # ── Anti-chatter hold: for NEXT step (1-step lag) ──
        # Only activate during active phase; during ramp, track E but don't hold.
        if self._warmup_phase == 'active':
            if self._climate_E_fast > self.climate_E_hold_on:
                self._climate_hold_active = True
            elif self._climate_E_fast < self.climate_E_hold_off:
                self._climate_hold_active = False
            # else: hysteresis band — hold state unchanged
        else:
            self._climate_hold_active = False


    # ================================================================= Rate governor (§10)

    def _rate_governor(self, lrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        eps = 1e-12
        if self._warmup_snaps_taken:
            noise_ref = max((self._noise_ref if self._noise_ref_seeded else self._snap_vol_thr), eps)
        else:
            noise_ref = max(self.vol_thr, eps)
        noise_intensity = self._rel_vol_ema / noise_ref

        k = self.gov_noise_sensitivity

        up_range = max(self.max_jump_up - 1.0, 0.0)
        self._gov_jump_up = max(1.0 + up_range / (1.0 + k * noise_intensity),
                                self._gov_jump_up_floor)

        down_range = max(self.max_jump_down - self._gov_jump_down_floor, 0.0)
        self._gov_jump_down = max(
            self._gov_jump_down_floor + down_range / (1.0 + k * noise_intensity),
            self._gov_jump_down_floor)

        floor = self.base_lr * 0.1
        low = np.maximum(self.prev_lrs * self._gov_jump_down, floor)
        high = self.prev_lrs * self._gov_jump_up

        governed = np.clip(lrs, low, high)
        hold_lr = ~np.isclose(lrs, governed, rtol=1e-3, atol=1e-12)
        return governed, hold_lr

    # ================================================================= Auto-gating (§6)

    def _update_auto_gating(self, u_sat: np.ndarray, hold_lr: np.ndarray,
                            lr_clamped_flag: bool, flip_ratio: float,
                            gaps_db: np.ndarray, calm_active: bool) -> None:
        if self.integrator_mode != "auto":
            self._gate_disable_reason_now = 0
            return

        a = self.gate_ema_alpha
        sat_val = float(np.mean(u_sat.astype(float)))
        # hold_val includes both governor hold AND ratio clamp
        hold_val = float(np.mean(hold_lr.astype(float)))
        if lr_clamped_flag:
            hold_val = max(hold_val, 1.0)  # clamp counts as "fully held"
        gap_val = float(np.mean(np.abs(gaps_db)))

        self._sat_ema = (1.0 - a) * self._sat_ema + a * sat_val
        self._hold_ema = (1.0 - a) * self._hold_ema + a * hold_val
        self._flip_ema = (1.0 - a) * self._flip_ema + a * flip_ratio
        self._gap_ema = (1.0 - a) * self._gap_ema + a * gap_val

        if self._gate_cooldown_remaining > 0:
            self._gate_cooldown_remaining -= 1

        was_enabled = self._i_enabled

        # Disable conditions — track reason as bitmask
        # bit 0: sat, bit 1: hold, bit 2: flip, bit 3: calm
        # Two fields:
        #   _gate_disable_reason_latched: set at disable, held until re-enable
        #   _gate_disable_reason_now: current active triggers (live diagnostic)

        reason_now = 0
        if self._sat_ema >= self.sat_off_thr:
            reason_now |= 1
        if self._hold_ema >= self.hold_off_thr:
            reason_now |= 2
        if self._flip_ema >= self.flip_off_thr:
            reason_now |= 4
        if calm_active:
            reason_now |= 8

        if self._i_enabled:
            # Only sat/hold/flip trigger full disable + cooldown.
            # Calm does NOT trigger disable here — it's already handled by
            # i_enabled_this_step = self._i_enabled and not calm_active.
            # This prevents calm from imposing a 50-step post-cooldown.
            disable_triggers = reason_now & 0x07  # mask out calm bit
            if disable_triggers:
                self._i_enabled = False
                self._gate_cooldown_remaining = self.gate_cooldown
                self._gate_disable_reason_latched = disable_triggers

        # Enable conditions: all EMAs below thresholds AND no calm
        if not self._i_enabled and self._gate_cooldown_remaining == 0:
            if (self._sat_ema < self.sat_on_thr and
                self._hold_ema < self.hold_on_thr and
                self._flip_ema < self.flip_on_thr and
                    not calm_active):
                self._i_enabled = True
                self._gate_disable_reason_latched = 0

        self._gate_disable_reason_now = reason_now

        if was_enabled != self._i_enabled:
            self._i_toggle_count += 1

    # ================================================================= Flip detector

    def _update_flip_ratio(self, gaps: np.ndarray) -> float:
        s = np.sign(gaps).astype(np.int8)
        s[np.abs(gaps) < self.gap_deadband] = 0

        both_definite = (s != 0) & (self.prev_gap_sign != 0)
        flips_mask = (s * self.prev_gap_sign) < 0

        if np.any(both_definite):
            flip_frac = float(np.sum(flips_mask & both_definite) / np.sum(both_definite))
        else:
            flip_frac = 0.0

        self.flip_hist.append(flip_frac)
        definite = s != 0
        self.prev_gap_sign = np.where(definite, s, self.prev_gap_sign)
        return float(np.mean(self.flip_hist)) if len(self.flip_hist) else 0.0

    # ================================================================= Adaptive thresholds

    def _update_adaptive_thresholds(self, rel_vol: float, d_mag_mean: float) -> None:
        a = self.stat_alpha

        self._rel_vol_ema = (1.0 - a) * self._rel_vol_ema + a * rel_vol
        dev = rel_vol - self._rel_vol_ema
        self._rel_vol_var_ema = (1.0 - a) * self._rel_vol_var_ema + a * (dev * dev)
        vol_std = math.sqrt(max(self._rel_vol_var_ema, 1e-18))
        raw_thr = self._rel_vol_ema + self.vol_thr_k * vol_std
        self.vol_thr = float(np.clip(raw_thr, self.vol_thr_clamp[0], self.vol_thr_clamp[1]))

        self._d_mag_ema = (1.0 - a) * self._d_mag_ema + a * d_mag_mean
        dev_d = d_mag_mean - self._d_mag_ema
        self._d_mag_var_ema = (1.0 - a) * self._d_mag_var_ema + a * (dev_d * dev_d)
        d_std = math.sqrt(max(self._d_mag_var_ema, 1e-18))
        raw_d_thr = self._d_mag_ema + self.d_calm_thr_k * d_std
        self.d_calm_thr = float(np.clip(raw_d_thr, self.d_calm_thr_clamp[0], self.d_calm_thr_clamp[1]))

    # ================================================================= Predictor

    def _update_predictor(self, shock: float, raw_d_mean: float,
                          integral_mean: float,
                          freeze_slow: bool = False) -> None:
        raw = np.array([shock, raw_d_mean, integral_mean], dtype=np.float64)
        fa = self.osc_alpha
        sa = self.osc_base_alpha

        if not self._pred_seeded:
            self._pred_fast = raw.copy()
            self._pred_slow = raw.copy()
            self._pred_seeded = True
        else:
            self._pred_fast = (1.0 - fa) * self._pred_fast + fa * raw
            # Freeze slow baseline during calm/shock to prevent
            # intervention-induced drift (same principle as adaptive thresholds).
            # Fast tracks current reality; slow stays at pre-intervention level.
            if not freeze_slow:
                self._pred_slow = (1.0 - sa) * self._pred_slow + sa * raw

        cap = max(self.osc_score_cap, 1e-6)
        min_bl = self.osc_min_baseline
        scores = np.zeros(self._pred_n, dtype=np.float64)

        for i in range(self._pred_n):
            slow_abs = abs(self._pred_slow[i])
            if slow_abs < min_bl:
                scores[i] = 0.0
            else:
                rel_diff = (self._pred_fast[i] - self._pred_slow[i]) / slow_abs
                scores[i] = float(np.clip(rel_diff / cap, 0.0, 1.0))

        self._pred_scores = scores.copy()
        self.osc_risk = float(np.clip(np.mean(scores), 0.0, 1.0))

        active_mask = np.array([abs(self._pred_slow[i]) >= min_bl
                                for i in range(self._pred_n)])
        n_active = int(np.sum(active_mask))

        if n_active >= 2:
            active_scores = scores[active_mask]
            mu = float(np.mean(active_scores))
            mad = float(np.mean(np.abs(active_scores - mu)))
            self.osc_conf = float(np.clip(1.0 - 2.0 * mad, 0.0, 1.0))
        else:
            self.osc_conf = 0.0
