# GravBalancer

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
