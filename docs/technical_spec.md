# GravBalancer v10.6.8 — Technical Specification

## What Is This?

GravBalancer is an **adaptive learning rate controller** for multi-player optimization.

In training setups where multiple neural networks compete or cooperate — most commonly GANs (Generative Adversarial Networks), but also multi-agent RL, game-theoretic optimization, and any scenario with N≥2 interacting loss functions — the relative learning rates between players critically affect stability and convergence. Too fast for one player, too slow for another, and training collapses. Traditional fixes are manual: pick a fixed ratio (TTUR), or hand-tune schedules. GravBalancer replaces that with an automatic, online controller.

**Analogy:** An orchestra conductor. The conductor doesn't play any instrument (GravBalancer doesn't modify gradients, architectures, or loss functions). It listens to how each section sounds (stress proxies) and adjusts their relative volume (learning rates) to keep the ensemble balanced.

## What It Does

**Input:** N scalar "stress proxies" — one per player, each step. These are real-valued scalars that reflect each player's current training difficulty. For GANs, this is typically the per-player loss; for other setups, any metric that goes up when a player is struggling and down when it's comfortable.

**Output:** N learning rates — one per player, each step.

**Core behavior:** If player A's proxy is high relative to player B's, GravBalancer increases A's learning rate and decreases B's, helping A "catch up." The total learning rate budget stays anchored to the user-supplied `base_lr`.

## What It Does NOT Do (By Design)

- **Does not interpret proxy semantics.** It doesn't know what "loss" means or which direction is "good." It manages dynamics only: relative imbalance → corrective LR adjustment.
- **Does not replace hyperparameter tuning.** It manages *relative* LR balance, not *absolute* learning rate choice. A bad `base_lr` is still bad.
- **Does not modify gradients, architectures, or objectives.** It is a pure outer-loop controller.
- **Does not guarantee convergence.** If the underlying optimization is fundamentally ill-posed, no LR controller can fix it. GravBalancer aims to *not make things worse* and to *help when balance matters*.
- **Does not require scenario-specific tuning.** All thresholds derive from observed system behavior during warmup. The same defaults work across cooperative, adversarial, asymmetric, nonstationary, and multi-player (N>2) regimes.

## Architecture Overview

### Pipeline (strict execution order, every step)

```
proxies → stats → gaps → steering (damper + integrator) → quiet-gate →
calm/shock → stability throttle → ratio clamp → governor →
comfort throttle → jump-clip → integrator update → auto-gating → diagnostics
```

Every component runs every step. There are no early-exit branches. The pipeline is deterministic given the same input sequence.

### Signal Flow

```
stress proxies  ──→  [Stats EMA]  ──→  gaps (who's struggling?)
                                          │
                                          ▼
                              ┌──── [Steering] ────┐
                              │  Damper (P-term)    │
                              │  Integrator (I-term)│
                              └─────────────────────┘
                                          │
                                     u_total (command)
                                          │
                        ┌─── [Safety layers: calm, throttle] ───┐
                        │     Shock detection                    │
                        │     Mechanical overshoot               │
                        │     Stability throttle                 │
                        └────────────────────────────────────────┘
                                          │
                              ┌──── [Walls (contract)] ────┐
                              │  Ratio clamp (max_ratio)    │
                              │  Rate governor (jump limits) │
                              │  Final jump-clip             │
                              └─────────────────────────────┘
                                          │
                                    learning rates out
```

## Core Components

### 1. Statistics (EMA Tracking)

For each player, the controller maintains exponential moving averages of the proxy value (μ) and its volatility (σ). A global reference scale (`ref_scale_ema`) tracks the overall magnitude, making all downstream computations scale-invariant.

**Window:** `stat_window=39` → α ≈ 0.05. This means the controller "remembers" approximately the last 40 steps.

### 2. Gaps ("Who is out of balance?")

```
gaps[i] = (μ[i] − consensus) / volatility_floor[i]
```

The consensus is a volatility-weighted mean of all player means. Gaps are positive when a player is above consensus (struggling more), negative when below. The volatility floor prevents division by near-zero in quiet signals.

### 3. Steering Signal (Dynamic Gaps)

**Default mode (`gaps_mode="dynamic"`):** Instead of using raw gaps (which carry absolute-level offsets from things like BCE loss structure), the controller steers from *derivatives* of gaps — baseline-removed, self-normalized:

```
d_filtered = EMA(gaps[t] − gaps[t−1])          # PT1-filtered change
d_centered = d_filtered − slow_baseline         # remove drift
z = d_centered / σ_d                            # self-normalize to z-scores
```

This makes the controller react to *changes in imbalance*, not absolute level.

**Deadband:** A calibrated deadband (`dyn_deadband_z`, auto-set from warmup noise percentile) removes the noise floor. Only z-scores exceeding the deadband generate commands.

### 4. Damper (P-term)

The damper converts deadband-clipped z-scores into a command:

```
targets = tanh(damper_k × gaps_db)    # saturating nonlinearity
u_p = (1−β) × u_p + β × targets      # EMA smoothing
u_p = clip(u_p, ±u_cap)              # hard limit
```

The tanh prevents extreme commands. The EMA smoothing (`beta_u=0.3`) prevents jitter. The hard cap (`u_cap`) is derived from `max_ratio` to guarantee that steering alone cannot violate the ratio contract.

**Anti-reverse:** When the flip detector reports oscillation, sign-conflicting targets are damped by `reverse_damp=0.3×`.

**Quiet-gate (B):** When the system is quiet (`rel_vol_ema` low), the damper command is scaled down by a smooth factor (`quiet_factor ∈ [quiet_gate_min, 1.0]`), preventing noise-induced jitter from reaching the output. A z-score bypass (`quiet_gate_bypass_z=3.0`) ensures real events still get through. Only active in `dynamic` gaps mode (z-score semantics).

### 5. Integrator (I-term)

The integrator accumulates persistent imbalance that the damper alone cannot correct:

```
delta_i = ki × gaps_db
delta_i -= mean(delta_i)    # zero-sum projection (reduces cross-coupling)
u_i += delta_i
u_i *= (1 − relax)         # continuous decay toward zero
```

**Variant C anti-windup:**
- Real LR walls (governor hold, ratio clamp) → global freeze: `delta_i = 0` entirely
- Steering saturation (`|u_total| ≈ u_cap`) → conditional: only block components that push *further* into saturation, allow components that reduce saturation

This replaces the prior global freeze that blocked the integrator ~90% of the time.

**Auto-gating (§6):** In `auto` mode, the integrator is automatically disabled when conditions suggest it would be counterproductive (high saturation, high hold rate, flipping). Hysteresis with cooldown prevents rapid toggling.

### 6. Safety Systems

**Shock detection:** Measures "surprise" — deviation of current proxies from their EMA, normalized by reference scale. Three detection paths:

| Path | Condition | Meaning |
|------|-----------|---------|
| BYPASS | `shock > 1.5× threshold` | Unmistakable single-step shock |
| SUSTAINED | `shock_ema > threshold` | Persistent elevation (not a blip) |
| NORMAL | `shock > threshold` AND system noisy | Standard detection in active regime |

Quiet systems (low `rel_vol_ema`) suppress single-step noise blips but still fire on large shocks (BYPASS) and sustained shifts (SUSTAINED).

**Calm:** When shock, mechanical overshoot, or oscillation predictor triggers, the system enters "calm mode" for `calm_len` steps. Different causes get different treatment:
- Shock (cause 1): hard stop — zero out u_p, u_i, u_total
- Mechanical overshoot (cause 2/3): exponential dampening of u_total
- Preemptive/oscillation (cause 4): exponential dampening of u_total

**Mechanical overshoot detector:** Monitors whether the controller's correction is proportionally too large relative to the gap's closing speed. Fires when the controller might overshoot.

**Oscillation predictor:** Compares fast vs slow EMA of (shock, d_derivative, integral magnitude) to detect acceleration patterns that precede oscillation. Fires preemptively with hysteresis.

### 7. Throttle

**Stability throttle (≤1.0):** Applied *before* ratio clamp. Reduces all LRs proportionally when noise is high:

```
stability = 1 / (1 + k_stab × max(0, noise_norm − deadband))
```

**Comfort throttle (≥1.0):** Applied *after* governor. Slightly boosts LRs when the system has been quiet for `boost_min_duration` steps. Gated by: no calm, no saturation, no wall hits, no shock. Headroom-limited so it never provokes the final jump-clip.

### 8. Contract Walls

These are the hard constraints that guarantee bounded behavior.

**Ratio clamp:** `max(LRs) / min(LRs) ≤ max_ratio` (default 1.618). Mean-anchored: the clamp preserves average LR.

**Rate governor:** Limits step-to-step LR change: `prev × jump_down ≤ LR ≤ prev × jump_up`. The window narrows adaptively when noise is high.

**Final jump-clip:** After comfort boost, LRs are clipped again to governor bounds. This is the last wall in the pipeline and cannot be bypassed.

### 9. Warmup

For the first `warmup_steps` (default 1000), the controller outputs `base_lr` for all players while collecting statistics:
- EMA means and volatilities stabilize
- Shock threshold calibrated from observed noise
- Volatility thresholds snapped from observed behavior
- Deadband auto-calibrated from z-score percentile (p90 of warmup noise)
- d_sigma (per-player derivative volatility) seeded

No steering, no integrator, no governor during warmup. This is calibration-only.

## Key Design Principles

### Scale Invariance
All thresholds are derived from observed system behavior, never hardcoded to specific loss scales. The same defaults work whether proxies are in [0, 1] or [0, 10000].

### No Scenario-Specific Tuning
The controller uses the same parameters for cooperative (aligned goals), adversarial (opposing goals), asymmetric (unequal noise), nonstationary (role swaps), and multi-player (N>2) scenarios. Tested with a synthetic plant harness across all five regimes.

### Conservative by Default
The controller prefers inaction over overcorrection. Deadbands, quiet-gates, governor rate limits, and calm modes all bias toward "don't make things worse."

### Transparent Diagnostics
Every step emits 60 diagnostic keys covering every pipeline stage. Shock path, calm cause, quiet factor, governor clip amounts, integrator effectiveness — all observable without modifying the code.

## API

### Constructor
```python
gb = GravBalancer(
    n_players=2,          # number of players (≥2)
    base_lr=2e-4,         # base learning rate (applied to all players equally)
    warmup_steps=1000,    # steps of pure observation before steering begins
    profile='competitive' # 'competitive' (GAN default) or 'cooperative'
)
```

All other parameters have sensible defaults. See `__init__` signature for full list.

### Per-Step Call
```python
lrs = gb.adjust(proxy_0, proxy_1, ..., proxy_N)  # returns list of N learning rates
diagnostics = gb.last_metrics                      # dict with 60 keys
```

### Reset
```python
gb.reset_state(force=True)  # full state reset for new experiment
```

## Diagnostic Keys Reference

| Key | Type | Meaning |
|-----|------|---------|
| `u_p` | array | Damper (P-term) command per player |
| `u_i` | array | Integrator (I-term) accumulation per player |
| `u_sat_steer` | array | Per-player steering saturation flag |
| `wall_lr_any` | bool | Any real LR wall (governor/clamp) active |
| `wall_any` | bool | Any wall including steering saturation |
| `i_enabled` | bool | Integrator active this step |
| `calm` | bool | Calm mode active |
| `calm_cause` | int | 0=none, 1=shock, 2=mech(low-vol), 3=mech/legacy, 4=preempt |
| `shock_path` | int | 0=none, 1=bypass, 2=sustained, 3=normal |
| `shock_ema` | float | Smoothed shock level |
| `shock_thr` | float | Current shock threshold |
| `quiet_factor` | float | Quiet-gate damping (0.05–1.0) |
| `dyn_deadband_z` | float | Effective deadband (z-score units) |
| `dyn_deadband_z_base` | float | Warmup-calibrated deadband base |
| `z_max` | float | Max absolute z-score across players |
| `stability_factor` | float | Stability throttle (0–1) |
| `gov_jump_up` | float | Current governor up-limit |
| `gov_jump_down` | float | Current governor down-limit |
| `lrs_raw` | array | LRs before governor |
| `lrs_gov` | array | LRs after governor, before comfort |
| `lrs_final` | array | Final output LRs |
| `prev_lrs` | array | Previous step's final LRs |

## Version History (v10.6.x)

| Version | Key Changes |
|---------|-------------|
| v10.6.6 | Diagnostic improvements, calm cause tracking |
| v10.6.7 | Variant C anti-windup, shock calm differentiation, sustained shock path |
| v10.6.8 | Warmup percentile deadband (A), quiet-gate (B), shock params to `__init__`, N-invariant z-collection (p90), `dyn_deadband_z_max` cap, `quiet_gate_min` floor, governor jump metrics, shock_path enum, level-mode guard for quiet-gate |

## File Inventory

| File | Purpose |
|------|---------|
| `grav_balancer_v10_6_8.py` | Controller implementation (~1190 lines) |
| `gravbalancer_v10_6_8_benchmark.ipynb` | 2D Gaussian mixture GAN benchmark (4 methods, 2 regimes) |
| `gravbalancer_harness.py` | Synthetic plant harness (5 scenarios, no NN needed) |
