"""Drop-in replacement for the `lap` package.

On this machine, Windows Application Control blocks the `lap` package's compiled
`_lapjv` DLL (unsigned, freshly downloaded via pip). Ultralytics' tracker imports
`lap` unconditionally at module load time, so we register a pure-Python/SciPy
stand-in in sys.modules before ultralytics.trackers is ever imported.
"""

import sys
import types

import numpy as np
from scipy.optimize import linear_sum_assignment as _scipy_lsa


def lapjv(cost_matrix, extend_cost=True, cost_limit=None):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost_matrix.shape
    x = np.full(n, -1, dtype=int)
    y = np.full(m, -1, dtype=int)
    if n == 0 or m == 0:
        return 0.0, x, y

    row_ind, col_ind = _scipy_lsa(cost_matrix)
    total = 0.0
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost_limit is not None and cost > cost_limit:
            continue
        x[r] = c
        y[c] = r
        total += cost
    return total, x, y


def install():
    if "lap" in sys.modules:
        return
    module = types.ModuleType("lap")
    module.__version__ = "0.5.13-scipy-shim"
    module.lapjv = lapjv
    sys.modules["lap"] = module
