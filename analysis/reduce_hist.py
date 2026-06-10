# analysis/reduce_hist.py
# v1.0 — Streaming reduction of sweep hist npz dumps + self-contained
# HTML report with curves (same philosophy as harness_report.html:
# one file, inline images, overwritten each run; history lives in git).
#
# Pipeline citizen (repair_plan addendum d/e discipline):
#   - deterministic, cheap, versioned cache with provenance
#   - processes npz files ONE AT A TIME (lazy np.load) — RAM stays in MBs
#   - mini self-test: python analysis/reduce_hist.py --selftest
#
# Usage:
#   python analysis/reduce_hist.py benchmark_output_10.9.2
#   → <dir>/analysis_cache.json   (compact per-epoch aggregates + events)
#   → <dir>/sweep_curves_report.html
#
# Collapse-event EXTRACTION here is descriptive (spike segments), NOT the
# pre-registered collapse criterion — that decision is the owner's, pending
# full multi-seed data (rule 2026-06-09: no conclusions on unproven guesses).

import base64
import io
import json
import os
import re
import sys

import numpy as np

REDUCER_VERSION = "1.0"

# Toy grid geometry (must match notebook config; validated against data
# by _validate_grid on the first file that has final_fake).
GRID_SIZE = 10
GRID_SPACING = 2.0
MODE_RADIUS = 0.5
MIN_COUNT_CKPT = 5      # on ckpt clouds (3000 pts) — dynamics shape only;
                        # NOT comparable to final 20k-protocol numbers.


def mode_centers(grid_size=GRID_SIZE, spacing=GRID_SPACING):
    half = (grid_size - 1) / 2.0
    xs = (np.arange(grid_size) - half) * spacing
    return np.array([(x, y) for x in xs for y in xs])


def _validate_grid(pts):
    """Sanity: a healthy cloud should sit near assumed centers."""
    c = mode_centers()
    d = np.linalg.norm(pts[:2000, None, :] - c[None, :, :], axis=2).min(axis=1)
    return float(np.median(d))


def coverage_stats(pts, centers, radius=MODE_RADIUS, min_count=MIN_COUNT_CKPT):
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    nearest = d.min(axis=1)
    counts = (d < radius).sum(axis=0)
    covered = int((counts >= min_count).sum())
    off_manifold = float((nearest > radius).mean())   # «усы/складки»: доля
    return covered, off_manifold                      # точек вдали от ВСЕХ мод


def spike_segments(loss_max_per_step, spe, k=10.0):
    """Descriptive explosion events: |loss| > k × running-median, grouped
    into segments separated by ≥1 epoch. Returns [(ep_start, ep_end, amp)]."""
    med = np.median(loss_max_per_step)
    if med <= 0 or not np.isfinite(med):
        return []
    idx = np.where(loss_max_per_step > k * med)[0]
    if len(idx) == 0:
        return []
    segs, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > spe:
            segs.append((s, a))
            s = b
    segs.append((s, idx[-1]))
    return [(round(a / spe, 2), round(b / spe, 2),
             float(loss_max_per_step[a:b + 1].max())) for a, b in segs]


def per_epoch(arr, spe, fn=np.mean):
    n = len(arr) // spe
    if n == 0:
        return np.asarray([fn(arr)]) if len(arr) else np.zeros(0)
    return np.asarray([fn(arr[i * spe:(i + 1) * spe]) for i in range(n)])


def reduce_one(path, spe):
    """One npz → compact dict (everything per-epoch or per-checkpoint)."""
    d = np.load(path)
    lD, lG = d['loss_D'], d['loss_G']
    mx = np.maximum(np.abs(lD), np.abs(lG))
    out = {
        'steps': int(len(lD)),
        'loss_D_ep': per_epoch(lD, spe).tolist(),
        'loss_G_ep': per_epoch(lG, spe).tolist(),
        'loss_max_ep': per_epoch(mx, spe, np.max).tolist(),
        'lr_G_ep': per_epoch(d['lr_G'], spe).tolist(),
        'spike_segments': spike_segments(mx, spe),
    }
    for k_src, k_dst, fn in [
        ('grav_climate_lr_meta', 'lr_meta_ep', np.mean),
        ('grav_climate_E_slow', 'E_slow_ep', np.mean),
        ('grav_panic_factor', 'panic_ep', np.min),
        ('grav_calm', 'calm_ep', np.mean),
        ('grav_trip_active', 'trip_ep', np.mean),
        ('grav_authority_factor', 'auth_ep', np.mean),
        ('grav_dyn_deadband_z_eff', 'db_ep', np.mean),
        ('grav_ramp_degraded', 'degraded', None),
        ('div_lambda', 'lambda_ep', np.mean),
        ('div_lambda_pid', 'lambda_pid_ep', np.mean),
        ('div_loss', 'div_loss_ep', np.mean),
    ]:
        if k_src in d.files:
            a = d[k_src]
            if a.size == 0:
                continue
            if fn is None:
                out[k_dst] = float(a[-1])
            else:
                out[k_dst] = per_epoch(a, spe, fn).tolist()

    # checkpoint clouds → coverage / off-manifold dynamics
    if 'ckpt_pts' in d.files and d['ckpt_pts'].size:
        centers = mode_centers()
        eps_, cov_, off_ = [], [], []
        for e, pts in zip(d['ckpt_epochs'], d['ckpt_pts']):
            cov, off = coverage_stats(np.asarray(pts), centers)
            eps_.append(float(e)); cov_.append(cov); off_.append(round(off, 4))
        out['ckpt_epochs'] = eps_
        out['ckpt_coverage'] = cov_          # protocol: 3000 pts, r=0.5, min5
        out['ckpt_off_manifold'] = off_
    if 'final_fake' in d.files and d['final_fake'].size:
        out['grid_median_dist'] = round(_validate_grid(d['final_fake']), 3)
    d.close()
    return out


def reduce_dir(out_dir, spe=976):
    """All hist_seed*_v*/hist_*.npz under out_dir → cache dict."""
    cache = {'reducer_version': REDUCER_VERSION, 'spe': spe, 'runs': {}}
    pat = re.compile(r'hist_seed(\d+)_v([\w.]+)$')
    for sub in sorted(os.listdir(out_dir)):
        m = pat.match(sub)
        if not m:
            continue
        seed, ver = m.group(1), m.group(2)
        cache.setdefault('sweep_version', ver)
        subdir = os.path.join(out_dir, sub)
        for f in sorted(os.listdir(subdir)):
            m2 = re.match(r'hist_(.+?)_(grav_div|baseline|grav|div)\.npz$', f)
            if not m2:
                continue
            lr_str, mode = m2.group(1), m2.group(2)
            key = f"seed{seed}|{lr_str}|{mode}"
            try:
                cache['runs'][key] = reduce_one(os.path.join(subdir, f), spe)
                print(f"  reduced {key} ({cache['runs'][key]['steps']} steps, "
                      f"{len(cache['runs'][key]['spike_segments'])} spike segs)")
            except Exception as e:                      # noqa: BLE001
                print(f"  [!] {key}: {e}")
                cache['runs'][key] = {'error': str(e)}
    return cache


# ────────────────────────────────────────────────────────── report

MODES_ORDER = ['baseline', 'grav', 'div', 'grav_div']
COLORS = {'baseline': '#777777', 'grav': '#1f77b4',
          'div': '#2ca02c', 'grav_div': '#d62728'}


def _fig_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=95, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _lr_panel(cache, seed, lr_str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = {m: cache['runs'].get(f"seed{seed}|{lr_str}|{m}") for m in MODES_ORDER}
    runs = {m: r for m, r in runs.items() if r and 'error' not in r}
    if not runs:
        return None
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"seed {seed}  |  LR {lr_str}", fontsize=13)

    ax = axes[0]   # SURVIVAL: покрытие мод по чекпойнтам
    for m, r in runs.items():
        if 'ckpt_coverage' in r:
            ax.plot(r['ckpt_epochs'], r['ckpt_coverage'], marker='o', ms=3,
                    lw=1.2, color=COLORS[m], label=m)
    ax.set_ylabel("modes (ckpt, 3k pts)")
    ax.set_ylim(0, 102); ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)

    ax = axes[1]   # off-manifold: «усы/складки»
    for m, r in runs.items():
        if 'ckpt_off_manifold' in r:
            ax.plot(r['ckpt_epochs'], r['ckpt_off_manifold'], marker='.',
                    lw=1.0, color=COLORS[m], label=m)
    ax.set_ylabel("off-manifold frac"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)

    ax = axes[2]   # взрывы: max|loss| per epoch, лог-шкала + сегменты
    for m, r in runs.items():
        e = np.arange(1, len(r['loss_max_ep']) + 1)
        ax.semilogy(e, np.maximum(r['loss_max_ep'], 1e-3), lw=1.0,
                    color=COLORS[m], label=m)
        for (a, b, amp) in r['spike_segments']:
            ax.axvspan(a, b + 0.3, color=COLORS[m], alpha=0.10)
    ax.set_ylabel("max|loss| /ep (log)"); ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)

    ax = axes[3]   # управление: lr_meta (grav-армы), λ (div-армы), panic
    for m, r in runs.items():
        e = np.arange(1, len(r['loss_max_ep']) + 1)
        if 'lr_meta_ep' in r:
            ax.plot(e[:len(r['lr_meta_ep'])], r['lr_meta_ep'], lw=1.2,
                    color=COLORS[m], label=f"{m}: lr_meta")
        if 'panic_ep' in r:
            p = np.asarray(r['panic_ep'])
            if (p < 0.999).any():
                ax.plot(e[:len(p)], p, lw=0.8, ls=':', color=COLORS[m],
                        label=f"{m}: panic_min")
        if 'lambda_ep' in r and m in ('div', 'grav_div'):
            lam = np.asarray(r['lambda_ep'])
            ax.plot(e[:len(lam)], np.clip(lam / max(lam.max(), 1e-9), 0, 1),
                    lw=0.8, ls='--', color=COLORS[m],
                    label=f"{m}: λ/{lam.max():.2f}")
    ax.set_ylabel("control [0..1]"); ax.set_xlabel("epoch")
    ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def render_report(cache, out_path):
    import time
    seeds = sorted({k.split('|')[0] for k in cache['runs']})
    def _lr_key(s):
        try:
            return (0, float(s.replace('e-0', 'e-')))
        except ValueError:
            return (1, 0.0)
    lrs = sorted({k.split('|')[1] for k in cache['runs']}, key=_lr_key)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>GravBalancer sweep curves</title>",
        "<style>body{font-family:sans-serif;margin:20px;background:#fafafa}"
        "h2{border-bottom:2px solid #888}table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ccc;padding:3px 8px}th{background:#eee}"
        "img{max-width:100%;border:1px solid #ddd;margin:6px 0}</style></head><body>",
        f"<h1>Sweep curves — v{cache.get('sweep_version','?')}</h1>"
        f"<p>generated {time.strftime('%Y-%m-%d %H:%M')} | reducer v{REDUCER_VERSION} | "
        f"перезаписывается (история = git). Ckpt-покрытие: 3000 точек, r=0.5, "
        f"min5 — НЕ сравнивать с финальным 20k-протоколом.</p>",
        "<h2>Сводка: события взрывов</h2><table><tr><th>run</th>"
        "<th>spike segs</th><th>max amp</th><th>degraded</th><th>сегменты (ep)</th></tr>",
    ]
    for key in sorted(cache['runs']):
        r = cache['runs'][key]
        if 'error' in r:
            parts.append(f"<tr><td>{key}</td><td colspan=4>ERR {r['error']}</td></tr>")
            continue
        segs = r['spike_segments']
        amp = max((s[2] for s in segs), default=0)
        seg_s = '; '.join(f"{a}–{b}" for a, b, _ in segs[:6]) + ('…' if len(segs) > 6 else '')
        parts.append(f"<tr><td>{key}</td><td>{len(segs)}</td><td>{amp:.1e}</td>"
                     f"<td>{int(r.get('degraded', 0))}</td><td>{seg_s}</td></tr>")
    parts.append("</table>")
    for seed in seeds:
        parts.append(f"<h2>{seed}</h2>")
        for lr_str in lrs:
            fig = _lr_panel(cache, seed.replace('seed', ''), lr_str)
            if fig is not None:
                parts.append(f"<h3>{seed} | LR {lr_str}</h3>")
                parts.append(f"<img src='data:image/png;base64,{_fig_b64(fig)}'/>")
    parts.append("</body></html>")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    return out_path


# ────────────────────────────────────────────────────────── selftest / cli

def _selftest():
    import tempfile
    spe = 50
    with tempfile.TemporaryDirectory() as td:
        sub = os.path.join(td, 'hist_seed1_v0.0')
        os.makedirs(sub)
        n = spe * 4
        rng = np.random.RandomState(0)
        loss = np.abs(1 + 0.1 * rng.randn(n)); loss[120:125] = 500.0  # спайк
        pts = rng.randn(3, 100, 2) * 6
        np.savez_compressed(
            os.path.join(sub, 'hist_5e-03_grav.npz'),
            loss_D=loss, loss_G=loss * 0.5, lr_G=np.full(n, 1e-3),
            grav_climate_lr_meta=np.linspace(1, 0.5, n),
            grav_ramp_degraded=np.zeros(n),
            ckpt_epochs=np.array([1., 2., 3.]), ckpt_pts=pts,
            final_fake=rng.randn(500, 2) * 6)
        cache = reduce_dir(td, spe=spe)
        r = cache['runs']['seed1|5e-03|grav']
        assert r['steps'] == n and len(r['spike_segments']) == 1
        assert abs(r['spike_segments'][0][0] - 120 / spe) < 0.1
        assert len(r['ckpt_coverage']) == 3 and 'lr_meta_ep' in r
        render_report(cache, os.path.join(td, 'r.html'))
        assert os.path.getsize(os.path.join(td, 'r.html')) > 10000
    print("selftest OK")


def main():
    if '--selftest' in sys.argv:
        _selftest(); return 0
    if len(sys.argv) < 2:
        print(__doc__); return 1
    out_dir = sys.argv[1]
    spe = int(sys.argv[2]) if len(sys.argv) > 2 else 976
    cache = reduce_dir(out_dir, spe=spe)
    cp = os.path.join(out_dir, 'analysis_cache.json')
    with open(cp, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f"cache: {cp} ({os.path.getsize(cp)//1024} KB)")
    rp = render_report(cache, os.path.join(out_dir, 'sweep_curves_report.html'))
    print(f"report: {rp} ({os.path.getsize(rp)//1024} KB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
