"""
B-011: JacobianPartition.nnz() / fill_ratio() on a non-diagonal sparse pattern.

On 3e78ed96 nnz() read `sm._rows`, which no Submat defines, and raised AttributeError. Fixed on om4 main by
40d5fceb (nnz sums `sm.data_size`). Expected: nnz 14 (5 + 4 + 5 listings, minus the repeated (0, 0)).
"""

import gc
from typing import Any
import numpy as np
from pydantic import model_validator
import om4.api as om
from om4.derivatives import jacobian_partition as jp


class Tri(om.ExplicitComponent):
    """Non-diagonal pattern: y_i depends on x_i and x_{i-1}, plus a dense-ish column."""

    @model_validator(mode="before")
    @classmethod
    def _b(cls, data: Any):
        n = 5
        rows = [i for i in range(n)] + [i for i in range(1, n)] + [i for i in range(n)]
        cols = [i for i in range(n)] + [i - 1 for i in range(1, n)] + [0] * n
        data["inputs"] = {"x": om.InputVar(val=np.ones(n), units=None)}
        data["outputs"] = {"y": om.OutputVar(val=np.ones(n), units=None)}
        data["partials"] = [om.PartialsSpec(of="y", wrt="x", rows=rows, cols=cols)]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs["x"]
        y = x.copy()
        y[1:] += x[:-1]
        y += x[0]
        outputs["y"] = y

    def compute_partials(self, inputs, partials):
        n = 5
        partials["y", "x"] = np.ones(n + (n - 1) + n)


for fmt in ("dense", "sparse"):
    p = om.Problem(model=om.Group(subsystems={"c": Tri()}, linear_solver=om.DirectSolver(assembled_format=fmt)))
    p.run_model()
    p.compute_totals(of=["c.y"], wrt=["c.x"])
    parts = [o for o in gc.get_objects() if isinstance(o, jp.JacobianPartition)]
    for o in parts:
        try:
            print(fmt, type(o).__name__, "nnz", o.nnz(), "fill_ratio", round(o.fill_ratio(), 3))
        except Exception as e:
            print(fmt, type(o).__name__, "ERR", type(e).__name__, e)
