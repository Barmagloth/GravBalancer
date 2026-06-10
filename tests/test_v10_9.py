# tests/test_v10_9.py
# Regression tests for the v10.9 repair release (docs/repair_plan_v1_0.md).
# Runnable standalone (python tests/test_v10_9.py) or via pytest.
# numpy-only; DiversitySentinel tests are skipped if torch is unavailable.

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grav_balancer import GravBalancer  # noqa: E402


def _run(gb, n, rng, gen=None):
    for t in range(n):
        vals = gen(t, rng) if gen else (1.0 + 0.05 * rng.randn(gb.N))
        gb.adjust(*np.abs(vals))


# ───────────────────────────── Tier 2: input contract

def test_negative_proxy_raises():
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=1000)
    try:
        gb.adjust(-0.5, 0.3)
    except ValueError as e:
        assert "non-negative" in str(e)
        return
    raise AssertionError("negative proxy was accepted")


def test_tiny_negative_proxy_clamped_not_fatal():
    # Amendment П1: FP-noise negatives must NOT kill a long run.
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gb.adjust(-1e-12, 1.0)
    assert any("clamped" in str(x.message) for x in w), "first clamp must warn"
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        gb.adjust(-1e-12, 1.0)
    assert not any("clamped" in str(x.message) for x in w2), \
        "subsequent clamps must be silent"


def test_nan_proxy_raises():
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=1000)
    try:
        gb.adjust(float('nan'), 0.3)
    except ValueError:
        return
    raise AssertionError("NaN proxy was accepted")


# ───────────────────────────── Tier 1: API / lifecycle

def test_dead_params_removed():
    for kw in ({'k_stab': 0.0}, {'noise_db': 0.3}, {'stab_min': 0.2}):
        try:
            GravBalancer(n_players=2, **kw)
        except TypeError:
            continue
        raise AssertionError(f"dead param accepted: {kw}")


def test_reset_state_restores_constructor_values():
    # Force a degraded run (impossible convergence criteria), then reset.
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=100,
                      convergence_cv_thr=-1.0, dyn_deadband_z=1.0)
    rng = np.random.RandomState(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(gb, 400, rng)
    assert gb._ramp_degraded, "run should have exited degraded"
    assert gb.dyn_deadband_z == gb.dyn_deadband_z_max  # pinned during run
    assert gb._ramp_steps_total == 200                  # doubled during run

    gb.reset_state(force=True)
    assert gb.dyn_deadband_z == 1.0, "dyn_deadband_z leaked through reset"
    assert gb.dyn_deadband_z_base == 1.0, "dyn_deadband_z_base leaked"
    assert gb._ramp_steps_total == 100, "_ramp_steps_total leaked"
    assert gb._degraded_reason == ''


def test_reset_state_equivalent_to_fresh():
    kwargs = dict(n_players=2, base_lr=1e-3, warmup_steps=100)
    gb = GravBalancer(**kwargs)
    rng = np.random.RandomState(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(gb, 500, rng)
    gb.reset_state(force=True)
    fresh = GravBalancer(**kwargs)

    skip = {'last_metrics'}
    for k, v in fresh.__dict__.items():
        if k in skip:
            continue
        got = gb.__dict__[k]
        if isinstance(v, np.ndarray):
            assert np.array_equal(got, v), f"state leak in {k}: {got} != {v}"
        elif isinstance(v, float) and np.isnan(v):
            assert isinstance(got, float) and np.isnan(got), \
                f"state leak in {k}: {got} != NaN"
        elif isinstance(v, (int, float, bool, str)) or v is None:
            assert got == v, f"state leak in {k}: {got} != {v}"


# ───────────────────────────── Tier 3: warmup feasibility

def test_infeasible_explicit_window_raises():
    try:
        GravBalancer(n_players=2, warmup_steps=98, convergence_window=200)
    except ValueError as e:
        assert "impossible" in str(e)
        return
    raise AssertionError("arithmetically impossible warmup config accepted")


def test_autoscale_window_converges_short_warmup():
    # The exact config of the v10.8.x CIFAR sweeps (warmup_steps ~= 98):
    # used to ALWAYS exit degraded with deadband pinned at 3.5.
    gb = GravBalancer(n_players=2, base_lr=5e-3, warmup_steps=98)
    assert gb._convergence_window_auto
    rng = np.random.RandomState(0)
    _run(gb, 400, rng)
    assert gb._warmup_converged, "stationary signal must converge"
    assert not gb._ramp_degraded
    assert gb.dyn_deadband_z_base < gb.dyn_deadband_z_max, \
        "deadband must be calibrated, not pinned at max"


def test_default_warmup_keeps_old_window():
    # warmup_steps=1000 (repo benchmark default) → auto window stays 200,
    # preserving v10.8 behavior for correctly-sized warmups.
    gb = GravBalancer(n_players=2, warmup_steps=1000)
    assert gb.convergence_window == 200


def test_degraded_exit_warns():
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=100,
                      convergence_cv_thr=-1.0)
    rng = np.random.RandomState(1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _run(gb, 400, rng)
    assert gb._ramp_degraded
    assert gb._degraded_reason == 'timeout'
    assert any("DEGRADED" in str(x.message) for x in w), \
        "degraded exit must emit a warning"


def test_too_short_warmup_warns_at_init():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GravBalancer(n_players=2, warmup_steps=10)
    assert any("too short" in str(x.message) for x in w)


# ───────────────────────────── Tier 0: metrics freshness

def test_snap_step_metrics_fresh():
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=98)
    rng = np.random.RandomState(3)
    prev_phase = 'A'
    seen_transition = False
    for t in range(400):
        gb.adjust(*np.abs(1.0 + 0.05 * rng.randn(2)))
        if prev_phase in ('A', 'B') and gb._warmup_phase == 'ramp':
            assert gb.last_metrics.get('warmup_phase') == 'ramp', \
                "last_metrics stale on warmup snap step"
            seen_transition = True
        prev_phase = gb._warmup_phase
    assert seen_transition


# ───────────────────────────── Tier 4: core invariants

def test_zero_sum_invariant_n3():
    gb = GravBalancer(n_players=3, base_lr=1e-3, warmup_steps=98)
    rng = np.random.RandomState(2)
    checked = 0
    for t in range(900):
        v = 1.0 + 0.05 * rng.randn(3)
        if t > 400 and t % 97 < 5:
            v = v + 0.3 * np.array([1.0, -0.5, -0.5])
        gb.adjust(*np.abs(v))
        m = gb.last_metrics
        if not m.get('warmup'):
            expected = gb.base_lr * m['panic_factor']
            drift = abs(float(np.mean(m['lrs_raw'])) - expected) / expected
            assert drift < 1e-9, f"zero-sum drift {drift} at step {t}"
            checked += 1
    assert checked > 0


def test_project_zero_sum_box_unit():
    gb = GravBalancer(n_players=4, warmup_steps=1000)
    rng = np.random.RandomState(5)
    for _ in range(200):
        u = rng.randn(4) * rng.uniform(0.01, 2.0)
        cap = rng.uniform(0.01, 0.5)
        v = gb._project_zero_sum_box(u, cap)
        assert abs(float(np.mean(v))) < 1e-12
        assert np.all(np.abs(v) <= cap + 1e-12)
    # cap = 0 → zeros
    assert np.all(gb._project_zero_sum_box(np.array([1.0, -1.0, 0.5]), 0.0) == 0)


def test_no_ramp_saturation_artifact():
    gb = GravBalancer(n_players=2, base_lr=1e-3, warmup_steps=500)
    rng = np.random.RandomState(7)
    ramp_start = None
    sat_steps = 0
    max_E = 0.0
    for t in range(1, 2600):
        base = np.array([0.8, 1.2])
        if gb._warmup_phase in ('ramp', 'active'):
            if ramp_start is None:
                ramp_start = t
            k = min((t - ramp_start) / 50.0, 1.0)
            base = base + np.array([0.4 * k, -0.4 * k])
        gb.adjust(*np.abs(base + 0.05 * rng.randn(2)))
        m = gb.last_metrics
        if not m.get('warmup') and gb._warmup_phase == 'ramp':
            sat_steps += int(np.mean(m['u_sat_steer']) > 0)
            max_E = max(max_E, m['climate_E'])
    assert sat_steps == 0, f"ramp saturation artifact: {sat_steps} steps"
    assert max_E < 2.0, f"climate_E inflated during ramp: {max_E}"


# ───────────────────────────── v10.9.1: climate winsorization (К1)

def test_climate_default_is_v109_ratchet():
    # (v10.9.2) К1 default OFF: the unwinsorized ratchet is the de-facto
    # emergency brake (seed-42 2e-2: lr_meta 0.058 -> survived; К1 -> died).
    gb = GravBalancer(n_players=2, warmup_steps=1000)
    assert gb._climate_E_winsor_mult == 0.0
    gb._climate_E_slow = 1.0
    gb._update_climate(np.array([8000.0, 8000.0]), 1.0, 0.0, 0.0, 0.0)
    assert gb._climate_E_slow > 10, "default must keep the fast emergency ratchet"


def test_climate_winsor_optin_single_spike_bounded():
    gb = GravBalancer(n_players=2, warmup_steps=1000, climate_E_winsor_mult=3.0)
    gb._climate_E_slow = 1.0
    gb._update_climate(resid_pp=np.array([8000.0, 8000.0]), shock_thr=1.0,
                       wall_rate_ema=0.0, u_sat_frac=0.0, clamp_frac=0.0)
    assert gb._climate_E > 100, "raw E (diagnostics) must still see the spike"
    assert gb._climate_E_slow < 1.02, \
        f"E_slow input must be winsorized, got {gb._climate_E_slow}"


def test_climate_winsor_optin_sustained_storm_still_brakes():
    gb = GravBalancer(n_players=2, warmup_steps=1000, climate_E_winsor_mult=3.0)
    for _ in range(4000):
        gb._update_climate(np.array([5.0, 5.0]), 1.0, 0.0, 0.0, 0.0)
    assert gb._climate_E_slow > 3.0, "sustained storm must still raise E_slow"
    assert gb._climate_lr_meta < 0.4, "sustained storm must still brake lr_meta"


def test_ds_lambda_preclamp_logged():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: torch unavailable)")
        return
    import torch
    from diversity_sentinel import make_sentinel_for_toy
    ds = make_sentinel_for_toy(warmup_steps=1)
    fake, real = torch.randn(32, 2), torch.randn(32, 2)
    ds.compute(fake, real)
    for _ in range(3):
        ds.step(1.0, 32)
    _, info = ds.compute(fake, real)
    assert 'div/lambda_pid_preclamp' in info


# ───────────────────────────── Harness integration

def test_harness_no_nan_keys():
    from harness import run_all
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_all(verbose=False)
    for name, r in results.items():
        k = r['kpis']
        assert k['no_nan_inf'], f"{name}: NaN keys {k['nan_keys']}"
        assert k['comfort_clipped_ok'], f"{name}: comfort_clipped failed"


# ───────────────────────────── DiversitySentinel (torch optional)

def test_ds_batch_mismatch_raises():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: torch unavailable)")
        return
    from diversity_sentinel import make_sentinel_for_toy
    ds = make_sentinel_for_toy()
    import torch
    fake = torch.randn(16, 2)
    real = torch.randn(8, 2)
    try:
        ds.compute(fake, real)
    except ValueError as e:
        assert "same size" in str(e)
        return
    raise AssertionError("batch mismatch was accepted")


def test_ds_cfg_not_shared():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: torch unavailable)")
        return
    from diversity_sentinel import DiversitySentinel
    a = DiversitySentinel()
    b = DiversitySentinel()
    assert a.cfg is not b.cfg, "default cfg shared between instances"


# ----------------------------- runner

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
