"""
B-003: check_partials double-counts duplicate (row, col) pairs in the reference Jacobian.

A sparse PartialsSpec may list the same (row, col) more than once; the analytic values are
summed (as in OM3), and totals use the sum correctly. check_partials' FD/CS reference is
built on the declared pattern too, but there each duplicate gets the full FD value, so the
reference entry is multiplied by its multiplicity and a correct component is reported wrong.
"""

from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om


class Dup(om.ExplicitComponent):
    """y = x * [3, 1, 1], with the (0, 0) entry declared twice (values 1 and 2, summing to 3)."""

    @model_validator(mode="before")
    @classmethod
    def _build(cls, data: Any):
        data["inputs"] = {"x": om.InputVar(val=np.ones(3), units=None)}
        data["outputs"] = {"y": om.OutputVar(val=np.ones(3), units=None)}
        data["partials"] = [om.PartialsSpec(of="y", wrt="x", rows=[0, 0, 1, 2], cols=[0, 0, 1, 2])]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = inputs["x"] * np.array([3.0, 1.0, 1.0])

    def compute_partials(self, inputs, partials):
        partials["y", "x"] = np.array([1.0, 2.0, 1.0, 1.0])


p = om.Problem(model=om.Group(subsystems={"c": Dup()}))
p.set_val("c.x", np.array([1.0, 2.0, 3.0]))
p.run_model()

data = p.check_partials()["c"][("y", "x")]
print("analytic J diag :", np.diag(data["J"]), "  (expected [3, 1, 1])")
print("reference Jref  :", np.round(np.diag(data["Jref"]), 6), "  (expected [3, 1, 1])")
print("totals (fwd)    :", np.diag(p.compute_totals(of=["c.y"], wrt=["c.x"], mode="fwd").todense()))
print("totals (rev)    :", np.diag(p.compute_totals(of=["c.y"], wrt=["c.x"], mode="rev").todense()))
print("reported wrong? :", bool(p.check_partials(only_incorrect=True)))
