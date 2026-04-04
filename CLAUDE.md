# GravBalancer — Project Handoff

## Permissions

This project lives under `R:\Projects\`. Permissions are configured in:
```
R:\Projects\.claude\settings.local.json
```
That file is the **source of truth** for allowed tools/actions across all projects in `R:\Projects\`.

## What This Is

GravBalancer is an **adaptive learning rate controller** for N-player optimization (GANs, multi-agent RL, game-theoretic training). It observes per-player stress proxies (typically losses) and adjusts learning rates to keep the system balanced.

Think of it as an orchestra conductor: doesn't play any instrument, just adjusts relative volume.

## Repository Structure

```
grav_balancer.py          # Core controller (v10.8, ~2190 lines)
diversity_sentinel.py     # Anti-mode-collapse module (v2.1, ~615 lines)
harness.py                # Synthetic plant test harness (5 scenarios)
docs/technical_spec.md    # Architecture reference (written at v10.6.8)
examples/
  benchmark_sweep.ipynb   # LR sweep benchmark (toy_2d + CIFAR-10)
README.md                 # Public-facing docs
```

## Architecture at a Glance

### grav_balancer.py — The Controller

**Input:** N stress proxies (scalars, one per player, each step).
**Output:** N learning rates.

Pipeline (strict order):
```
stats → gaps → steering (damper P-term + integrator I-term) → calm/shock →
authority × stress × ramp → ratio clamp → governor →
comfort → jump-clip → climate (lr_meta + anti-chatter) → diagnostics
```

Five safety layers (inner → outer):
1. **Authority** — scales steering cap based on noise level (continuous)
2. **Control Stress** — asymmetric EMA of event rates → authority squeeze
3. **Tripwire** — short intervention (10 steps) + cooldown (60 steps)
4. **Panic** — rare brake on base_lr for systemic distress
5. **Climate** — energy-based lr_meta for chronic storm detection (τ~1000)

Key invariants:
- Zero-sum steering: `mean(lrs) = base_lr × panic_factor` always
- Ratio contract: `max(lrs)/min(lrs) ≤ max_ratio` (default 1.618)
- Conservative by default: deadbands, quiet-gate, governor — biased toward inaction

Warmup: three-phase convergence-based (A: observe → B: converge → ramp: quadratic soft-start).

### diversity_sentinel.py — Anti-Mode-Collapse

Separate module, operates orthogonally to GravBalancer:
- GB controls **how fast** players learn (LR)
- Sentinel controls **what** the generator optimizes (adds diversity loss)

Uses radial + sliced Wasserstein distances. PID-controlled lambda (log-space).
One-way bridge: Sentinel reads GB distress level, backs off during emergencies.

**Critical contract:** GravBalancer proxies must come from MAIN loss only (not diversity loss), preventing feedback loops.

### harness.py — Synthetic Test Harness

Five scenarios without neural networks:
1. Cooperative (2 players, aligned goals)
2. Adversarial symmetric (2 players, anti-correlated)
3. Adversarial asymmetric (2 players, one 3× noisier)
4. Nonstationary (role swap at midpoint)
5. Multi-player (4 players, mixed coop+adversarial)

## API Quick Reference

```python
from grav_balancer import GravBalancer

gb = GravBalancer(n_players=2, base_lr=2e-4, warmup_steps=1000)

# Each training step:
lrs = gb.adjust(proxy_g, proxy_d)   # returns [lr_g, lr_d]
diag = gb.last_metrics               # ~80 diagnostic keys

# Reset for new experiment:
gb.reset_state(force=True)
```

## Legacy Archive

Full version history (v10.5 through v10.8, ~20 versions, all notebooks, CIFAR-10 data, benchmark results) is preserved in:
```
R:\Projects\GB_old\
```

## Dependencies

- `numpy` (core controller — no torch/GPU needed)
- `torch` (diversity_sentinel only)
- `matplotlib`, `torchvision`, `torchmetrics` (benchmark notebook only)

## GitHub

https://github.com/Barmagloth/GravBalancer
