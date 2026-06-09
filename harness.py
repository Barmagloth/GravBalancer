"""
GravBalancer — Multi-Scenario Synthetic Plant Harness (v10.9)

Purpose: verify controller behavior across diverse regimes.
NOT for tuning parameters — for checking "does it break?"

Scenarios:
  1. Cooperative (2 players, aligned goals)
  2. Adversarial symmetric (2 players, anti-correlated, equal noise)
  3. Adversarial asymmetric (2 players, one much noisier/stronger)
  4. Nonstationary / distribution shift mid-run
  5. Multi-player (N=4, mixed cooperative+adversarial)

Each scenario is a synthetic "plant" that produces stress proxies
as a function of LRs and internal state. No real neural nets.

v10.9 changes:
  - Removed stale metric key 'stability_factor' (gone since v10.7.0) which
    made the no_nan_inf hard invariant fail on EVERY scenario, masking the
    real NaN check. NaN check is now per-key (reports offending keys).
  - Removed k_stab=0.0 from the GravBalancer call (dead param since v10.7.0;
    its original intent "throttle off" had silently inverted into "authority
    on at default strength"). Throttle intent is now explicit: run_all() is
    executed for BOTH debug_freeze="none" (authority active, true defaults)
    and debug_freeze="no_throttle" (pure steering) — both are informative.
  - Collects v10.9 observability keys: authority_factor, panic_factor,
    climate_lr_meta, ramp_degraded, warmup_converged, dyn_deadband_z_eff.
  - Single self-contained HTML report (harness_report.html) with inline
    panels for every scenario × mode. Overwritten on each run by design:
    the report is a pure function of the code; history lives in git.
    Pass keep=True (CLI: --keep) to also save a timestamped copy.
"""

import base64
import io
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

from grav_balancer import GravBalancer


# ================================================================
# Synthetic plants
# ================================================================

class SyntheticPlant:
    """Base class. Subclass and implement step()."""
    def __init__(self, n_players: int, seed: int = 42):
        self.N = n_players
        self.rng = np.random.RandomState(seed)
        self.t = 0

    def step(self, lrs: np.ndarray) -> np.ndarray:
        """Given current LRs, return stress proxies for each player."""
        raise NotImplementedError

    def reset(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.t = 0


class CooperativePlant(SyntheticPlant):
    """
    Scenario 1: Both players benefit from balanced LRs.
    Stress = base_stress + noise + penalty for LR imbalance.
    Expected: controller keeps ratio ≈ 1, u_total_zm ≈ 0.
    """
    def __init__(self, seed=42):
        super().__init__(n_players=2, seed=seed)
        self.base = np.array([0.5, 0.5])

    def step(self, lrs):
        self.t += 1
        noise = self.rng.randn(2) * 0.02
        # Both players stressed equally, slight penalty for imbalance
        ratio = max(lrs) / max(min(lrs), 1e-12)
        penalty = 0.1 * (ratio - 1.0)
        stress = self.base + noise + penalty
        return np.clip(stress, 0.01, 10.0)


class AdversarialSymmetricPlant(SyntheticPlant):
    """
    Scenario 2: Anti-correlated stress (GAN-like).
    When player 0 does well (low stress), player 1 does worse, and vice versa.
    Equal noise scale.
    """
    def __init__(self, seed=42):
        super().__init__(n_players=2, seed=seed)

    def step(self, lrs):
        self.t += 1
        # Base oscillation: anti-correlated
        phase = 0.3 * np.sin(self.t * 0.01)
        base = np.array([0.7 + phase, 1.3 - phase])
        noise = self.rng.randn(2) * 0.05
        return np.clip(base + noise, 0.01, 10.0)


class AdversarialAsymmetricPlant(SyntheticPlant):
    """
    Scenario 3: One player much noisier and stronger.
    Player 1 has 3x noise and higher base stress.
    This is where "pretty" controllers usually die.
    """
    def __init__(self, seed=42):
        super().__init__(n_players=2, seed=seed)

    def step(self, lrs):
        self.t += 1
        noise_0 = self.rng.randn() * 0.03
        noise_1 = self.rng.randn() * 0.09  # 3x noisier
        # Player 1 consistently stronger
        base = np.array([0.5 + noise_0, 1.5 + noise_1])
        # LR feedback: higher LR reduces stress slightly
        lr_effect = (lrs - np.mean(lrs)) * 0.5
        return np.clip(base - lr_effect, 0.01, 10.0)


class NonstationaryPlant(SyntheticPlant):
    """
    Scenario 4: Distribution shift at t=500.
    First half: player 0 dominant.
    Second half: player 1 dominant (roles swap).
    Tests baseline removal, shock/calm, adaptation.
    """
    def __init__(self, n_steps=1000, seed=42):
        super().__init__(n_players=2, seed=seed)
        self.shift_at = n_steps // 2

    def step(self, lrs):
        self.t += 1
        noise = self.rng.randn(2) * 0.04
        if self.t < self.shift_at:
            base = np.array([0.6, 1.2])
        else:
            # Sudden role swap
            base = np.array([1.2, 0.6])
        return np.clip(base + noise, 0.01, 10.0)


class MultiPlayerPlant(SyntheticPlant):
    """
    Scenario 5: N=4 players.
    Players 0,1 cooperative pair. Players 2,3 adversarial pair.
    Cross-group: mild coupling.
    """
    def __init__(self, seed=42):
        super().__init__(n_players=4, seed=seed)

    def step(self, lrs):
        self.t += 1
        noise = self.rng.randn(4) * 0.03
        phase = 0.2 * np.sin(self.t * 0.015)
        # Cooperative pair (0,1): similar stress
        base_01 = 0.8 + noise[:2]
        # Adversarial pair (2,3): anti-correlated
        base_23 = np.array([0.7 + phase, 1.3 - phase]) + noise[2:]
        # Mild cross-group coupling
        cross = 0.05 * (np.mean(lrs[:2]) - np.mean(lrs[2:]))
        base = np.concatenate([base_01 + cross, base_23 - cross])
        return np.clip(base, 0.01, 10.0)


# ================================================================
# Harness runner
# ================================================================

# Scalar/per-player metrics collected from last_metrics each post-warmup step.
COLLECTED_KEYS = [
    'u_sat_steer', 'lr_wall_final', 'wall_lr_any', 'wall_any',
    'comfort_active', 'comfort_clipped', 'comfort_limited_by_headroom',
    'calm', 'calm_cause', 'i_enabled', 'gate_disable_reason',
    'toggle_count', 'z_max', 'ratio_max_over_mean',
    'ratio_mean_over_min', 'headroom_min', 'noise_norm',
    'authority_factor', 'panic_factor', 'climate_lr_meta',
    'ramp_degraded', 'warmup_converged', 'dyn_deadband_z_eff',
    'shock_path', 'quiet_factor',
    'dyn_deadband_z_base', 'dyn_deadband_z', 'trip_active',
]


def run_scenario(plant, gb, n_steps=1000):
    """Run one scenario, collect step-level diagnostics."""
    history = defaultdict(list)
    lrs = np.full(plant.N, gb.base_lr)

    for t in range(n_steps):
        proxies = plant.step(lrs)
        lrs_out = gb.adjust(*proxies)
        lrs = np.array(lrs_out)

        m = gb.last_metrics
        if m.get('warmup'):
            continue

        # Collect key signals
        history['lrs'].append(lrs.copy())
        history['proxies'].append(proxies.copy())
        for key in COLLECTED_KEYS:
            val = m.get(key, float('nan'))
            if hasattr(val, '__len__'):
                val = float(np.mean(val))
            history[key].append(float(val))

        # Extra: gaps signal strength
        gs = m.get('gaps_steer', None)
        history['gaps_steer_abs'].append(float(np.mean(np.abs(gs))) if gs is not None else 0.0)
        gd = m.get('gaps_db', None)
        history['gaps_db_abs'].append(float(np.mean(np.abs(gd))) if gd is not None else 0.0)

        history['u_i_norm'].append(float(np.linalg.norm(gb.u_i)))
        history['u_p_norm'].append(float(np.linalg.norm(gb.u_p)))

        # Governor: raw jump vs governed jump (per-element abs)
        lrs_raw_v = m.get('lrs_raw', lrs)
        lrs_gov_v = m.get('lrs_gov', lrs)
        prev = m.get('prev_lrs', lrs)
        if hasattr(lrs_raw_v, '__len__') and hasattr(prev, '__len__'):
            prev_safe = np.maximum(np.array(prev), 1e-12)
            raw_jump_abs = np.abs(np.array(lrs_raw_v) / prev_safe - 1.0)
            gov_jump_abs = np.abs(np.array(lrs_gov_v) / prev_safe - 1.0)
            history['raw_jump_abs_mean'].append(float(np.mean(raw_jump_abs)))
            history['raw_jump_abs_max'].append(float(np.max(raw_jump_abs)))
            history['gov_jump_abs_mean'].append(float(np.mean(gov_jump_abs)))
        else:
            history['raw_jump_abs_mean'].append(0.0)
            history['raw_jump_abs_max'].append(0.0)
            history['gov_jump_abs_mean'].append(0.0)

    # Convert to arrays
    for k in history:
        history[k] = np.array(history[k])

    return dict(history)


def compute_kpis(history, scenario_name=""):
    """Compute invariant and management KPIs from history."""
    kpis = {}
    n = len(history.get('u_sat_steer', []))
    if n == 0:
        return {'error': 'no post-warmup steps'}

    # --- Hard invariants ---
    comfort_clipped = history.get('comfort_clipped', np.array([]))
    kpis['comfort_clipped_rate'] = float(np.mean(comfort_clipped)) if len(comfort_clipped) > 0 else 0.0
    kpis['comfort_clipped_ok'] = kpis['comfort_clipped_rate'] < 0.01

    # NaN/inf check — per key, so a stale metric name is reported BY NAME
    # instead of silently failing the whole invariant (the v10.8 harness
    # failed this check on every scenario because of the removed
    # 'stability_factor' key, and nobody could see why).
    nan_keys = []
    for k, v in history.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            if not np.isfinite(v).all():
                nan_keys.append(k)
    kpis['nan_keys'] = nan_keys
    kpis['no_nan_inf'] = (len(nan_keys) == 0)

    # --- Warmup outcome ---
    rd = history.get('ramp_degraded', np.array([]))
    kpis['ramp_degraded'] = bool(rd[-1] > 0.5) if len(rd) > 0 else False
    wc = history.get('warmup_converged', np.array([]))
    kpis['warmup_converged'] = bool(wc[-1] > 0.5) if len(wc) > 0 else False

    # --- Management KPIs ---
    kpis['mean_u_sat_steer'] = float(np.mean(history['u_sat_steer']))
    kpis['mean_wall_lr_any'] = float(np.mean(history['wall_lr_any']))
    kpis['mean_wall_any'] = float(np.mean(history['wall_any']))
    kpis['mean_lr_wall_final'] = float(np.mean(history['lr_wall_final']))
    kpis['mean_i_enabled'] = float(np.mean(history['i_enabled']))
    kpis['max_toggle_count'] = float(np.max(history['toggle_count']))
    kpis['mean_z_max'] = float(np.mean(history['z_max']))
    kpis['mean_ratio_max_over_mean'] = float(np.mean(history['ratio_max_over_mean']))

    # I actually working? (i_enabled AND not wall_lr_any)
    i_active = history['i_enabled'] * (1.0 - history['wall_lr_any'])
    kpis['i_effective_rate'] = float(np.mean(i_active))

    # u_i accumulation: nonzero?
    kpis['final_u_i_norm'] = float(history['u_i_norm'][-1]) if len(history['u_i_norm']) > 0 else 0.0
    kpis['mean_u_i_norm'] = float(np.mean(history['u_i_norm']))

    # Calm frequency and cause breakdown
    calm = history.get('calm', np.array([]))
    kpis['calm_rate'] = float(np.mean(calm)) if len(calm) > 0 else 0.0
    calm_cause = history.get('calm_cause', np.array([]))
    if len(calm_cause) > 0 and kpis['calm_rate'] > 0:
        calm_mask = calm > 0.5
        causes_when_calm = calm_cause[calm_mask]
        total_calm = max(len(causes_when_calm), 1)
        kpis['calm_cause_0_none'] = float(np.sum(causes_when_calm == 0)) / total_calm
        kpis['calm_cause_1_shock'] = float(np.sum(causes_when_calm == 1)) / total_calm
        kpis['calm_cause_2_mech_lowvol'] = float(np.sum(causes_when_calm == 2)) / total_calm
        kpis['calm_cause_3_mech_legacy'] = float(np.sum(causes_when_calm == 3)) / total_calm
        kpis['calm_cause_4_preempt'] = float(np.sum(causes_when_calm == 4)) / total_calm
    else:
        kpis['calm_cause_0_none'] = 0.0
        kpis['calm_cause_1_shock'] = 0.0
        kpis['calm_cause_2_mech_lowvol'] = 0.0
        kpis['calm_cause_3_mech_legacy'] = 0.0
        kpis['calm_cause_4_preempt'] = 0.0

    # Shock path breakdown (when shock_trigger fires)
    shock_path = history.get('shock_path', np.array([]))
    if len(shock_path) > 0 and np.any(shock_path > 0):
        sp_mask = shock_path > 0
        total_sp = max(np.sum(sp_mask), 1)
        kpis['shock_path_1_bypass'] = float(np.sum(shock_path == 1)) / total_sp
        kpis['shock_path_2_sustained'] = float(np.sum(shock_path == 2)) / total_sp
        kpis['shock_path_3_normal'] = float(np.sum(shock_path == 3)) / total_sp
    else:
        kpis['shock_path_1_bypass'] = 0.0
        kpis['shock_path_2_sustained'] = 0.0
        kpis['shock_path_3_normal'] = 0.0

    # Comfort on rate
    comfort_active = history.get('comfort_active', np.array([]))
    kpis['comfort_on_rate'] = float(np.mean(comfort_active)) if len(comfort_active) > 0 else 0.0

    # Dynamic gaps signal strength
    gaps_steer_abs = history.get('gaps_steer_abs', np.array([]))
    kpis['mean_gaps_steer_abs'] = float(np.mean(gaps_steer_abs)) if len(gaps_steer_abs) > 0 else 0.0
    gaps_db_abs = history.get('gaps_db_abs', np.array([]))
    kpis['mean_gaps_db_abs'] = float(np.mean(gaps_db_abs)) if len(gaps_db_abs) > 0 else 0.0

    # Governor dominance: how often do LR walls constrain?
    kpis['governor_dominance'] = kpis['mean_lr_wall_final']
    kpis['governor_dominance_ok'] = kpis['governor_dominance'] < 0.7

    # Governor jump analysis (proper abs metrics)
    raw_jump = history.get('raw_jump_abs_mean', np.array([]))
    raw_jump_max = history.get('raw_jump_abs_max', np.array([]))
    gov_jump = history.get('gov_jump_abs_mean', np.array([]))
    if len(raw_jump) > 0:
        kpis['gov_raw_jump_mean'] = float(np.mean(raw_jump))
        kpis['gov_raw_jump_p95'] = float(np.percentile(raw_jump, 95))
        kpis['gov_raw_jump_max'] = float(np.max(raw_jump_max))
        kpis['gov_output_jump_mean'] = float(np.mean(gov_jump))
        kpis['gov_clip_ratio'] = 1.0 - float(np.mean(gov_jump)) / max(float(np.mean(raw_jump)), 1e-12)
    else:
        kpis['gov_raw_jump_mean'] = 0.0
        kpis['gov_raw_jump_p95'] = 0.0
        kpis['gov_raw_jump_max'] = 0.0
        kpis['gov_output_jump_mean'] = 0.0
        kpis['gov_clip_ratio'] = 0.0

    # Authority / safety layer levels
    kpis['mean_authority'] = float(np.mean(history['authority_factor']))
    kpis['mean_panic_factor'] = float(np.mean(history['panic_factor']))
    kpis['min_climate_lr_meta'] = float(np.min(history['climate_lr_meta']))

    # Quiet-gate
    qf = history.get('quiet_factor', np.array([]))
    kpis['mean_quiet_factor'] = float(np.mean(qf)) if len(qf) > 0 else 1.0
    kpis['quiet_active_rate'] = float(np.mean(qf < 0.99)) if len(qf) > 0 else 0.0

    # Adaptive deadband calibration
    ddb = history.get('dyn_deadband_z_base', np.array([]))
    kpis['dyn_deadband_z_base'] = float(ddb[0]) if len(ddb) > 0 else 0.0

    return kpis


# ================================================================
# Main harness
# ================================================================

SCENARIOS = {
    'cooperative': {
        'plant_cls': CooperativePlant,
        'n_players': 2,
        'n_steps': 1000,
        'description': 'Both players aligned. Expect ratio≈1, low u_sat.',
    },
    'adversarial_sym': {
        'plant_cls': AdversarialSymmetricPlant,
        'n_players': 2,
        'n_steps': 1000,
        'description': 'Anti-correlated, equal noise. Classic balance test.',
    },
    'adversarial_asym': {
        'plant_cls': AdversarialAsymmetricPlant,
        'n_players': 2,
        'n_steps': 1000,
        'description': 'One player 3x noisier. Where controllers die.',
    },
    'nonstationary': {
        'plant_cls': NonstationaryPlant,
        'n_players': 2,
        'n_steps': 1000,
        'description': 'Role swap at t=500. Tests adaptation/shock.',
    },
    'multiplayer_4': {
        'plant_cls': MultiPlayerPlant,
        'n_players': 4,
        'n_steps': 1000,
        'description': 'N=4: coop pair + adversarial pair + cross-coupling.',
    },
}

# Both throttle intents are informative and are run by default:
#   "default"     — debug_freeze="none": authority/panic/comfort active,
#                   true library defaults (what a user gets out of the box).
#   "no_throttle" — debug_freeze="no_throttle": pure steering, safety
#                   multipliers frozen at 1.0 (what the original v10.6-era
#                   harness intended with its k_stab=0.0, before the dead
#                   param silently inverted that intent).
MODES = {
    'default': 'none',
    'no_throttle': 'no_throttle',
}


def run_all(base_lr=1e-3, warmup_steps=100, seed=42, verbose=True,
            debug_freeze='none'):
    """Run all scenarios with default GravBalancer params. Returns results dict."""
    results = {}

    for name, cfg in SCENARIOS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {name}: {cfg['description']}  [throttle={debug_freeze}]")
            print(f"{'='*60}")

        plant = cfg['plant_cls'](seed=seed)

        gb = GravBalancer(
            n_players=cfg['n_players'],
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            # === library defaults; only the steering signal source and the
            # === throttle intent are made explicit ===
            gaps_mode='dynamic',
            dyn_deadband_z=1.0,
            integrator_mode='auto',
            profile='competitive',
            debug_freeze=debug_freeze,
        )

        history = run_scenario(plant, gb, n_steps=cfg['n_steps'])
        kpis = compute_kpis(history, name)

        results[name] = {
            'history': history,
            'kpis': kpis,
            'config': cfg,
        }

        if verbose:
            print(f"\n  KPIs:")
            # Hard invariants
            print(f"    [HARD] comfort_clipped_rate = {kpis['comfort_clipped_rate']:.4f}"
                  f"  {'OK' if kpis['comfort_clipped_ok'] else 'FAIL'}")
            print(f"    [HARD] no_nan_inf           = {kpis['no_nan_inf']}"
                  f"  {'OK' if kpis['no_nan_inf'] else 'FAIL: ' + ','.join(kpis['nan_keys'])}")
            print(f"    warmup: converged={kpis['warmup_converged']} "
                  f"degraded={kpis['ramp_degraded']} "
                  f"db_base={kpis.get('dyn_deadband_z_base', 0):.3f}")
            # Management
            print(f"    mean(u_sat_steer)    = {kpis['mean_u_sat_steer']:.3f}")
            print(f"    mean(wall_lr_any)    = {kpis['mean_wall_lr_any']:.3f}  (I hard-stop)")
            print(f"    i_effective_rate     = {kpis['i_effective_rate']:.3f}  (I enabled AND not wall)")
            print(f"    mean_u_i_norm        = {kpis['mean_u_i_norm']:.4f}")
            print(f"    calm_rate            = {kpis['calm_rate']:.3f}")
            print(f"    governor_dominance   = {kpis['governor_dominance']:.3f}"
                  f"  {'OK' if kpis['governor_dominance_ok'] else 'WARN high'}")
            print(f"    mean(z_max)          = {kpis['mean_z_max']:.2f}")
            print(f"    authority/panic/meta = {kpis['mean_authority']:.3f} / "
                  f"{kpis['mean_panic_factor']:.3f} / {kpis['min_climate_lr_meta']:.3f}")
            print(f"    quiet: factor={kpis.get('mean_quiet_factor', 1):.3f} "
                  f"active={kpis.get('quiet_active_rate', 0)*100:.1f}%")

    return results


def print_summary(results, mode_name=''):
    """Print compact summary table."""
    title = f"MULTI-SCENARIO SUMMARY ({mode_name})" if mode_name else "MULTI-SCENARIO SUMMARY"
    print(f"\n{'='*100}")
    print(f"{title:^100}")
    print(f"{'='*100}")
    print(f"{'Scenario':<20} {'cc':>4} {'nan':>5} {'degr':>5} {'u_sat':>6} {'wlr':>6} "
          f"{'I_eff':>6} {'ui_n':>6} {'calm':>6} {'gov':>6} {'z_max':>6} "
          f"{'auth':>6} {'|gd|':>6}")
    print('-' * 100)

    all_pass = True
    for name, r in results.items():
        k = r['kpis']
        cc = 'OK' if k['comfort_clipped_ok'] else 'X'
        nn = 'OK' if k['no_nan_inf'] else 'X'
        dg = 'yes' if k.get('ramp_degraded') else 'no'
        if not k['comfort_clipped_ok'] or not k['no_nan_inf']:
            all_pass = False

        print(f"{name:<20} {cc:>4} {nn:>5} {dg:>5} {k['mean_u_sat_steer']:>6.3f} "
              f"{k['mean_wall_lr_any']:>6.3f} {k['i_effective_rate']:>6.3f} "
              f"{k['mean_u_i_norm']:>6.4f} {k['calm_rate']:>6.3f} "
              f"{k['governor_dominance']:>6.3f} {k['mean_z_max']:>6.2f} "
              f"{k['mean_authority']:>6.3f} {k['mean_gaps_db_abs']:>6.3f}")

    print('-' * 100)
    print(f"Hard invariants: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


# ================================================================
# HTML report (single self-contained file, inline panels)
# ================================================================

def _fig_to_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=95, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _scenario_panel(history, name, mode_name):
    """4-panel dynamics figure for one scenario run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(len(history['lrs']))
    lrs = history['lrs']           # (T, N)
    proxies = history['proxies']   # (T, N)
    N = lrs.shape[1] if lrs.ndim > 1 else 1

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"{name}  [{mode_name}]", fontsize=13)

    ax = axes[0]
    for i in range(N):
        ax.plot(t, proxies[:, i], lw=0.7, label=f"p{i}")
    ax.set_ylabel("proxies")
    ax.legend(loc='upper right', fontsize=8, ncol=min(N, 4))
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(N):
        ax.plot(t, lrs[:, i], lw=0.9, label=f"lr{i}")
    ax.set_ylabel("LR")
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8, ncol=min(N, 4))
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, history['u_p_norm'], lw=0.8, label='|u_p|')
    ax.plot(t, history['u_i_norm'], lw=0.8, label='|u_i|')
    ax.plot(t, history['gaps_db_abs'], lw=0.7, label='|gaps_db|')
    ax.plot(t, np.minimum(history['z_max'], 10.0), lw=0.5, alpha=0.6, label='z_max (cap 10)')
    ax.plot(t, history['dyn_deadband_z_eff'], lw=0.8, ls='--', label='deadband_eff')
    ax.set_ylabel("steering")
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, history['authority_factor'], lw=0.9, label='authority')
    ax.plot(t, history['panic_factor'], lw=0.9, label='panic_factor')
    ax.plot(t, history['climate_lr_meta'], lw=0.9, label='lr_meta')
    ax.plot(t, history['quiet_factor'], lw=0.7, alpha=0.7, label='quiet')
    calm = history.get('calm', np.zeros(len(t)))
    trip = history.get('trip_active', np.zeros(len(t)))
    if np.any(calm > 0.5):
        ax.fill_between(t, 0, 1.05, where=calm > 0.5, color='red', alpha=0.12, label='calm')
    if np.any(trip > 0.5):
        ax.fill_between(t, 0, 1.05, where=trip > 0.5, color='orange', alpha=0.12, label='trip')
    ax.set_ylabel("safety [0..1]")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("post-warmup step")
    ax.legend(loc='lower right', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


_KPI_ROWS = [
    ('warmup_converged', 'warmup converged'),
    ('ramp_degraded', 'degraded'),
    ('dyn_deadband_z_base', 'deadband base'),
    ('mean_u_sat_steer', 'u_sat mean'),
    ('mean_wall_lr_any', 'wall_lr mean'),
    ('i_effective_rate', 'I effective'),
    ('mean_u_i_norm', '|u_i| mean'),
    ('calm_rate', 'calm rate'),
    ('mean_z_max', 'z_max mean'),
    ('mean_authority', 'authority mean'),
    ('mean_panic_factor', 'panic mean'),
    ('min_climate_lr_meta', 'lr_meta min'),
]


def render_html_report(all_results, out_path='harness_report.html', keep=False):
    """One self-contained HTML: every scenario × mode, panels inline.

    Overwritten on every run BY DESIGN — the report is a pure function of
    the current code; history is reproducible from git. keep=True saves an
    additional timestamped copy.
    """
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>GravBalancer harness report</title>",
        "<style>body{font-family:sans-serif;margin:20px;background:#fafafa}"
        "h2{border-bottom:2px solid #888}table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ccc;padding:3px 8px}th{background:#eee}"
        ".fail{color:#b00;font-weight:bold}.ok{color:#080}"
        "img{max-width:100%;border:1px solid #ddd;margin:6px 0}</style></head><body>",
        f"<h1>GravBalancer harness report</h1>"
        f"<p>generated: {stamp} | grav_balancer v10.9 | overwritten each run (history = git)</p>",
    ]

    for mode_name, results in all_results.items():
        parts.append(f"<h2>throttle mode: {mode_name}</h2>")
        for name, r in results.items():
            k = r['kpis']
            inv = ('<span class="ok">PASS</span>'
                   if k['no_nan_inf'] and k['comfort_clipped_ok']
                   else f'<span class="fail">FAIL (nan: {", ".join(k.get("nan_keys", [])) or "-"};'
                        f' comfort_clipped={k["comfort_clipped_rate"]:.4f})</span>')
            parts.append(f"<h3>{name} — {inv}</h3>")
            rows = ''.join(
                f"<tr><th>{label}</th><td>{k.get(key, '')if not isinstance(k.get(key), float) else f'{k[key]:.4f}'}</td></tr>"
                for key, label in _KPI_ROWS)
            parts.append(f"<table>{rows}</table>")
            fig = _scenario_panel(r['history'], name, mode_name)
            parts.append(f"<img src='data:image/png;base64,{_fig_to_b64(fig)}'/>")

    parts.append("</body></html>")
    html = '\n'.join(parts)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    if keep:
        ts_path = out_path.replace('.html', time.strftime('_%Y%m%d_%H%M%S.html'))
        with open(ts_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return out_path, ts_path
    return out_path, None


def main(base_lr=1e-3, warmup_steps=100, seed=42, report=True, keep=False,
         verbose=True):
    """Run all scenarios in both throttle modes; render the HTML report."""
    all_results = {}
    all_pass = True
    for mode_name, freeze in MODES.items():
        results = run_all(base_lr=base_lr, warmup_steps=warmup_steps,
                          seed=seed, verbose=verbose, debug_freeze=freeze)
        all_pass &= print_summary(results, mode_name)
        all_results[mode_name] = results

    if report:
        try:
            path, kept = render_html_report(all_results, keep=keep)
            print(f"\nHTML report: {os.path.abspath(path)}"
                  + (f" (+ kept copy: {kept})" if kept else ""))
        except ImportError:
            print("\nmatplotlib not available — HTML report skipped.")

    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_results, all_pass


if __name__ == '__main__':
    keep_flag = '--keep' in sys.argv
    no_report = '--no-report' in sys.argv
    _, ok = main(report=not no_report, keep=keep_flag)
    sys.exit(0 if ok else 1)
