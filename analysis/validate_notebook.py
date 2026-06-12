#!/usr/bin/env python3
# analysis/validate_notebook.py
# v1.0 — Единый валидатор benchmark-ноутбука. Запускать ПЕРЕД каждым коммитом
# ноутбука и перед длинными прогонами:
#   python analysis/validate_notebook.py examples/benchmark_sweep.ipynb
#
# Родился из инцидентов 2026-06-10/11 (каждая проверка — закрытый класс багов):
#   1. ast.parse каждой код-ячейки            ← черновые ячейки («22 15»)
#   2. shadow-аудит protected-имён             ← nn = np.mean(...) убил torch.nn
#   3. СТРУКТУРНЫЕ AST-инварианты              ← дедент warnings-блока выкинул
#      (train_gan/elapsed/метрики в for-mode)    train_gan из цикла
#   4. exec-симуляция конфиг-ячейки (5 пресетов)← PRESET переопределялся ниже
#   5. путевые инварианты (версионирование)     ← коллизии hist_seed42
# Линтеры (pylint W0631 и т.п.) проверены на инциденте №3 — НЕ ловят его
# на склейке ячеек; потому инварианты, а не линтер.

import ast
import json
import sys

PROTECTED = {'nn', 'np', 'torch', 'F', 'os', 'io', 'json', 'time', 'math',
             'sys', 'plt', 'Counter', 'device', 'data_loader', 'data_tensor',
             'data_meta', 'steps_per_epoch', 'make_models', 'train_gan',
             'compute_all_metrics', 'seed_everything', 'probe_quality',
             'generate_report', 'DiversitySentinel', 'GravBalancer',
             'make_sentinel', 'capture_gif_frame_toy', 'capture_gif_frame_img',
             '_probe_cache', 'kid_from_feats', 'density_coverage',
             'sliced_w_np', 'radial_w_np'}
# ячейки, где определение этих имён легитимно (по содержимому)
DEF_MARKERS = ('import ', 'def ', '= make_data', '_probe_cache = {}',
               'device = ')

PRESETS = ['smoke_toy', 'smoke_cifar', 'toy_full', 'cifar_full', 'manual']


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main(path):
    raw = open(path, 'rb').read().decode('utf-8', errors='replace')
    nb, end = json.JSONDecoder().raw_decode(raw)
    if end < len(raw.rstrip()):
        print(f"  [warn] хвостовой мусор после JSON ({len(raw)-end}B) — stale view?")
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    code_all = '\n\n'.join(''.join(c['source']) for c in cells)

    # 1. синтаксис каждой ячейки
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            fail(f"ячейка {i} не парсится: {e}\n  head: {src.strip()[:60]!r}")
    print(f"OK 1: {len(cells)} код-ячеек парсятся")

    # 2. shadow-аудит (присваивания на верхнем уровне ячейки, вне функций)
    class TA(ast.NodeVisitor):
        def __init__(self): self.names = set()
        def visit_FunctionDef(self, n): pass
        visit_AsyncFunctionDef = visit_FunctionDef
        def _t(self, t):
            if isinstance(t, ast.Name): self.names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts: self._t(e)
        def visit_Assign(self, n):
            for t in n.targets: self._t(t)
            self.generic_visit(n)
        def visit_For(self, n):
            self._t(n.target); self.generic_visit(n)
    bad = []
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        v = TA(); v.visit(ast.parse(src))
        hits = v.names & PROTECTED
        if hits and not any(m in src for m in DEF_MARKERS):
            bad.append((i, sorted(hits)))
    if bad:
        fail(f"затирание protected-имён вне ячеек-определений: {bad}")
    print("OK 2: shadow-аудит чист")

    # 3. структурные инварианты свип-ячейки
    sw = next((''.join(c['source']) for c in cells
               if 'for mode in MODES' in ''.join(c['source'])), None)
    if sw is None:
        fail("свип-ячейка (for mode in MODES) не найдена")
    ok = False
    for n in ast.walk(ast.parse(sw)):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) \
                and n.target.id == 'mode':
            b = ast.unparse(n)
            if 'train_gan(mode=mode)' in b and 'compute_all_metrics' in b \
                    and 'elapsed = time.time() - t0' in b:
                for m_ in ast.walk(n):
                    if isinstance(m_, ast.If) and isinstance(m_.test, ast.Name) \
                            and m_.test.id == '_wl':
                        s_ = ast.unparse(m_)
                        if 'train_gan' in s_ or 'elapsed' in s_ \
                                or 'compute_all_metrics' in s_:
                            fail("конвейер затянут внутрь if _wl")
                ok = True
    if not ok:
        fail("train_gan/metrics/elapsed НЕ в теле for-mode (инцидент aab3a5a)")
    # сид-цикл существует и содержит свип
    if 'for SEED in SEEDS' not in sw:
        fail("сид-цикл отсутствует в свип-ячейке")
    print("OK 3: структурные инварианты конвейера")

    # 4. exec-симуляция конфиг-ячейки по всем пресетам
    cfg = next((''.join(c['source']) for c in cells
                if 'SWEEP_VERSION' in ''.join(c['source'])
                and 'PRESET' in ''.join(c['source'])), None)
    if cfg is None:
        fail("конфиг-ячейка с PRESET не найдена")
    expect = {
        'smoke_toy':  dict(DATASET='toy_2d', EPOCHS=4, SEEDS=[42]),
        'smoke_cifar': dict(DATASET='cifar10', EPOCHS=2, SEEDS=[42]),
        'toy_full':   dict(DATASET='toy_2d', EPOCHS=80, SEEDS=[42, 43, 44]),
        # SEEDS в cifar_full редактируются владельцем под конкретный прогон —
        # проверяем форму (непустой список int), не значение
        'cifar_full': dict(DATASET='cifar10', EPOCHS=300),
    }
    import re
    for p in PRESETS:
        g = {}
        exec(re.sub(r'PRESET = "[a-z_]+"', f'PRESET = "{p}"', cfg, count=1), g)
        for k, v in expect.get(p, {}).items():
            if g.get(k) != v:
                fail(f"preset {p}: {k}={g.get(k)!r}, ожидалось {v!r}")
        s = g.get('SEEDS')
        if not (isinstance(s, list) and s and all(isinstance(x, int) for x in s)):
            fail(f"preset {p}: SEEDS={s!r} — не непустой список int")
        assert g['IS_TOY'] == (g['DATASET'] == 'toy_2d')
        assert g['LR_G'] == g['LR_GRID'][0]
    print(f"OK 4: exec-симуляция {len(PRESETS)} пресетов")

    # 5. путевые инварианты (версионная изоляция артефактов)
    for marker, why in [
        ('benchmark_output_{SWEEP_VERSION}', 'OUTPUT_DIR без версии'),
        ('hist_seed{SEED}_v{SWEEP_VERSION}', 'hist-дамп без версии'),
        ('evolution_seed{SEED}', 'gif без сида'),
        ("sweep_metrics_seed{SEED}", 'metrics без сида'),
        ('warnings_seed{SEED}.log', 'warnings-лог отсутствует'),
    ]:
        if marker not in code_all:
            fail(f"путевой инвариант: {why} ({marker!r})")
    print("OK 5: путевые инварианты")
    print("\nVALIDATE: ALL OK")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'examples/benchmark_sweep.ipynb')
