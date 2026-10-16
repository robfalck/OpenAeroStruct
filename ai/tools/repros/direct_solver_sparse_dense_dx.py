"""
B-014: DirectSolver(assembled_format="sparse") still assembles dR/dx as a dense matrix.

PartialJacobian builds DxPartition without an `assemble=` format, so it defaults to "dense", and
DirectSolver.linearize calls jac.dx.ensure_assembled(). For a cycle with n states and n boundary
inputs that is an (n_states x n_inputs) dense array even though every partial is diagonal.
On the OAS aerostructural point (4x41 mesh, ~88k states) it is an 87527 x 92247 float64 array
(64 GB virtual), which makes compute_totals about 3x slower than OM3's sparse assembly.
"""

from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives import jacobian_partition as jp

N = 2000


class Link(om.ExplicitComponent):
    """out = 0.5 * other + x, all elementwise (diagonal partials)."""

    @model_validator(mode="before")
    @classmethod
    def _build(cls, data: Any):
        idx = list(range(N))
        data["inputs"] = {
            "other": om.InputVar(val=np.ones(N), units=None),
            "x": om.InputVar(val=np.ones(N), units=None),
        }
        data["outputs"] = {"out": om.OutputVar(val=np.ones(N), units=None)}
        data["partials"] = [
            om.PartialsSpec(of="out", wrt="other", rows=idx, cols=idx),
            om.PartialsSpec(of="out", wrt="x", rows=idx, cols=idx),
        ]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["out"] = 0.5 * inputs["other"] + inputs["x"]

    def compute_partials(self, inputs, partials):
        partials["out", "other"] = np.full(N, 0.5)
        partials["out", "x"] = np.ones(N)


for cls in (jp.DxPartition, jp.DuPartition):
    for name in ("_assemble_dense", "_assemble_sparse"):
        orig = getattr(cls, name)

        def wrapped(self, exec_node, orig=orig, cls=cls, name=name):
            r = orig(self, exec_node)
            print(f"{cls.__name__}.{name[1:]:16} shape={r.shape}")
            return r

        setattr(cls, name, wrapped)

coupled = om.Group(
    subsystems={"a": Link(), "b": Link()},
    connections=[om.Connection(src="a.out", tgt="b.other"), om.Connection(src="b.out", tgt="a.other")],
    nonlinear_solver=om.NonlinearBlockGS(maxiter=100, atol=1e-12, rtol=1e-30),
    linear_solver=om.DirectSolver(assembled_format="sparse"),
)
p = om.Problem(model=om.Group(subsystems={"coupled": coupled}))
p.run_model()
p.compute_totals(of=["coupled.a.out"], wrt=["coupled.a.x"])
print("expected: both partitions assembled sparse; DxPartition is assembled dense instead")
