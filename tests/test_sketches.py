"""Generated column sketches sit on the intended grid."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SKETCH_PY = Path(__file__).resolve().parents[1] / 'inputs' / 'sketches' / 'make_sketches.py'


def _mod():
    spec = importlib.util.spec_from_file_location('make_sketches', SKETCH_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_braced_columns_sit_on_the_edges():
    module = _mod()
    xs = module.column_centers(3, edge=True)
    assert xs[0] == module.LINE / 2
    assert xs[-1] == module.WIDTH - module.LINE / 2
    inset = module.column_centers(3, edge=False)
    assert inset[0] > xs[0]
    assert inset[-1] < xs[-1]
