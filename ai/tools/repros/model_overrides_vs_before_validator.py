"""Repro: Problem.model_overrides silently loses its effect on any system nested inside a
Group subclass whose own @model_validator(mode="before") regenerates `subsystems` from other
fields (a common and in fact unavoidable pattern for parametric Groups, used throughout the
OpenAeroStruct om4 port -- see e.g. AerostructGeometry, CoupledAS, CoupledPerformance, TubeGroup,
Geometry, SpatialBeamSetup/AssembleKGroup, VLMFunctionals, SpatialBeamFunctionals in
openaerostruct/integration/aerostruct_groups.py and friends).

Root cause: Problem._apply_model_overrides (om4/core/problem.py) does
    model_dict = self.model.model_dump()
    <patch model_dict in place at the target pathname>
    self.model = SpecManager.load(model_dict)
This re-validates the ENTIRE tree from the patched dict. Any ancestor Group whose
model_validator(mode="before") derives `subsystems` from its own fields (rather than trusting
whatever `subsystems` dict is already present in the input data) discards the patched
`subsystems` subtree wholesale -- including any override baked into it -- and rebuilds it
fresh without the override.

om4's own test suite (om4/core/tests/test_problem.py::test_wildcard_reaches_root_and_nested_groups)
only exercises plain `Group(subsystems={...})` instances with no such validator, so this gap is
untested upstream.

Run: pixi run python ai/tools/repros/model_overrides_vs_before_validator.py
"""

from typing import Any

from pydantic import model_validator

import om4.api as om


class PlainParent(om.Group):
    """A Group built the "normal" way: subsystems passed straight through."""


class DerivedParent(om.Group):
    """A Group that *computes* its subsystems from another field, the way nearly every
    composite Group in the OpenAeroStruct om4 port is written (parametrized by something like
    a `surface` dict, not by a raw `subsystems` mapping)."""

    n_children: int = 1

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        n = data.get("n_children", 1)
        # The key thing OAS's real Groups do: nest *another Group* below, not just leaf
        # components -- so an override targeting that grandchild group has somewhere to land.
        data["subsystems"] = {
            f"child{i}": om.Group(subsystems={"comp": om.ExecComp(exprs=["y=2*x"])}) for i in range(n)
        }
        return data


def check(model, label):
    p = om.Problem(model=model, model_overrides={"*": {"collapse_by": "group"}})
    outer = p.model.collapse_by
    inner = p.model.subsystems["inner"].collapse_by
    grandchild = p.model.subsystems["inner"].subsystems["child0"].collapse_by
    ok = outer == "group" and inner == "group" and grandchild == "group"
    print(
        f"{label}: outer={outer!r} inner={inner!r} grandchild(inner.child0)={grandchild!r} "
        f"-> {'OK' if ok else 'BUG'}"
    )


check(
    om.Group(
        subsystems={
            "inner": PlainParent(subsystems={"child0": om.Group(subsystems={"comp": om.ExecComp(exprs=["y=2*x"])})})
        }
    ),
    "plain Group  ",
)
check(om.Group(subsystems={"inner": DerivedParent(n_children=2)}), "derived Group")
