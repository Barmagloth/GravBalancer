"""
GravBalancer v10.6.6 — Multi-Scenario Synthetic Plant Harness

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
"""

import numpy as np
from collections import defaultdict
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
        for key in ['u_sat_steer', 'lr_wall_final', 'wall_lr_any', 'wall_any',
                     'comfort_active', 'comfort_clipped', 'comfort_limited_by_headroom',
                     'calm', 'calm_cause', 'i_enabled', 'gate_disable_reason',
                     'toggle_count', 'z_max', 'ratio_max_over_mean',
                     'ratio_mean_over_min', 'headroom_min', 'noise_norm',
                     'stability_factor', 'shock_path', 'quiet_factor',
                     'dyn_deadband_z_base', 'dyn_deadband_z']:
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

    # NaN/inf check
    has_nan = False
    for k, v in history.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            if not np.isfinite(v).all():
                has_nan = True
                break
    kpis['no_nan_inf'] = not has_nan

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

    # Stability: not constant 1.0 (meaning stability throttle is active)?
    kpis['mean_stability'] = float(np.mean(history['stability_factor']))

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


def run_all(base_lr=1e-3, warmup_steps=100, seed=42, verbose=True):
    """Run all scenarios with default GravBalancer params. Returns results dict."""
    results = {}

    for name, cfg in SCENARIOS.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  {name}: {cfg['description']}")
            print(f"{'='*60}")

        plant = cfg['plant_cls'](seed=seed)

        gb = GravBalancer(
            n_players=cfg['n_players'],
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            # === DEFAULTS (not tuned for any scenario) ===
            stat_window=39,
            d_filter_window=19,
            osc_window=24,
            osc_base_window=399,
            damper_k=1.0,
            beta_u=0.3,
            gaps_mode='dynamic',
            dyn_deadband_z=1.0,
            dyn_floor_min=1e-4,
            integrator_mode='auto',
            ki=0.02,
            max_jump_up=1.25,
            max_jump_down=0.9,
            k_stab=0.0,
            profile='competitive',
            osc_preempt_calm=True,
            osc_require_high_vol=True,
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
                  f"  {'✓' if kpis['comfort_clipped_ok'] else '✗ FAIL'}")
            print(f"    [HARD] no_nan_inf           = {kpis['no_nan_inf']}"
                  f"  {'✓' if kpis['no_nan_inf'] else '✗ FAIL'}")
            # Management
            print(f"    mean(u_sat_steer)    = {kpis['mean_u_sat_steer']:.3f}")
            print(f"    mean(wall_lr_any)    = {kpis['mean_wall_lr_any']:.3f}  (I hard-stop)")
            print(f"    mean(wall_any)       = {kpis['mean_wall_any']:.3f}  (incl u_sat)")
            print(f"    i_effective_rate     = {kpis['i_effective_rate']:.3f}  (I enabled AND not wall)")
            print(f"    mean_u_i_norm        = {kpis['mean_u_i_norm']:.4f}")
            print(f"    final_u_i_norm       = {kpis['final_u_i_norm']:.4f}")
            print(f"    calm_rate            = {kpis['calm_rate']:.3f}")
            print(f"    max_toggle_count     = {kpis['max_toggle_count']:.0f}")
            print(f"    governor_dominance   = {kpis['governor_dominance']:.3f}"
                  f"  {'✓' if kpis['governor_dominance_ok'] else '⚠ high'}")
            print(f"    gov: raw_jump(mean={kpis.get('gov_raw_jump_mean',0)*100:.2f}% "
                  f"p95={kpis.get('gov_raw_jump_p95',0)*100:.2f}% "
                  f"max={kpis.get('gov_raw_jump_max',0)*100:.1f}%) "
                  f"→ out_mean={kpis.get('gov_output_jump_mean',0)*100:.2f}% "
                  f"clip_ratio={kpis.get('gov_clip_ratio',0):.1%}")
            print(f"    mean(z_max)          = {kpis['mean_z_max']:.2f}")
            print(f"    mean(ratio_m/m)      = {kpis['mean_ratio_max_over_mean']:.3f}")
            print(f"    quiet: factor={kpis.get('mean_quiet_factor',1):.3f} "
                  f"active={kpis.get('quiet_active_rate',0)*100:.1f}% "
                  f"db_base={kpis.get('dyn_deadband_z_base',0):.3f}")

    return results


def print_summary(results):
    """Print compact summary table."""
    print(f"\n{'='*100}")
    print(f"{'MULTI-SCENARIO SUMMARY':^100}")
    print(f"{'='*100}")
    print(f"{'Scenario':<20} {'cc✓':>4} {'nan✓':>5} {'u_sat':>6} {'wlr':>6} "
          f"{'I_eff':>6} {'ui_n':>6} {'calm':>6} {'gov':>6} {'z_max':>6} "
          f"{'|gs|':>6} {'|gd|':>6}")
    print('-' * 100)

    all_pass = True
    for name, r in results.items():
        k = r['kpis']
        cc = '✓' if k['comfort_clipped_ok'] else '✗'
        nn = '✓' if k['no_nan_inf'] else '✗'
        if not k['comfort_clipped_ok'] or not k['no_nan_inf']:
            all_pass = False

        print(f"{name:<20} {cc:>4} {nn:>5} {k['mean_u_sat_steer']:>6.3f} "
              f"{k['mean_wall_lr_any']:>6.3f} {k['i_effective_rate']:>6.3f} "
              f"{k['mean_u_i_norm']:>6.4f} {k['calm_rate']:>6.3f} "
              f"{k['governor_dominance']:>6.3f} {k['mean_z_max']:>6.2f} "
              f"{k['mean_gaps_steer_abs']:>6.3f} {k['mean_gaps_db_abs']:>6.3f}")

    print('-' * 100)
    print(f"Hard invariants: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

    # Calm cause breakdown
    print(f"\n{'CALM CAUSE BREAKDOWN':^60}")
    print(f"{'Scenario':<20} {'calm%':>6} {'shock':>7} {'mch_lo':>7} {'mch/lg':>7} {'pre':>7}")
    print('-' * 60)
    for name, r in results.items():
        k = r['kpis']
        cr = k['calm_rate']
        if cr > 0.001:
            print(f"{name:<20} {cr*100:>5.1f}% "
                  f"{k.get('calm_cause_1_shock',0)*100:>6.1f}% "
                  f"{k.get('calm_cause_2_mech_lowvol',0)*100:>6.1f}% "
                  f"{k.get('calm_cause_3_mech_legacy',0)*100:>6.1f}% "
                  f"{k.get('calm_cause_4_preempt',0)*100:>6.1f}%")
        else:
            print(f"{name:<20} {cr*100:>5.1f}%     —       —       —       —")
    print()


if __name__ == '__main__':
    results = run_all(base_lr=1e-3, warmup_steps=100, seed=42)
    print_summary(results)
