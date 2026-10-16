from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.failure_exact import FailureExact
from openaerostruct.structures.failure_ks import FailureKS
from openaerostruct.structures.non_intersecting_thickness import NonIntersectingThickness
from openaerostruct.structures.tsaiwu_wingbox import TsaiWuWingbox
from openaerostruct.structures.vonmises_tube import VonMisesTube
from openaerostruct.structures.vonmises_wingbox import VonMisesWingbox
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface

_WINGBOX_STRESS_INPUTS = ["Qz", "J", "A_enc", "spar_thickness", "htop", "hbottom", "hfront", "hrear", "nodes", "disp"]


def _spatial_beam_functionals_kwargs(surface: Surface) -> dict:
    subs = {}
    use_composite = bool(surface.get("useComposite", False))
    stress_name = "tsaiwu_sr" if use_composite else "vonmises"

    fem_model_type = surface["fem_model_type"].lower()
    if fem_model_type == "tube":
        subs["thicknessconstraint"] = NonIntersectingThickness(
            surface=surface, promotes_inputs=["thickness", "radius"], promotes_outputs=["thickness_intersects"]
        )
        subs["vonmises"] = VonMisesTube(
            surface=surface, promotes_inputs=["radius", "nodes", "disp"], promotes_outputs=["vonmises"]
        )
    elif fem_model_type == "wingbox":
        stress_cls = TsaiWuWingbox if use_composite else VonMisesWingbox
        subs["tsaiwu_sr" if use_composite else "vonmises"] = stress_cls(
            surface=surface, promotes_inputs=_WINGBOX_STRESS_INPUTS, promotes_outputs=[stress_name]
        )
    else:
        raise NameError("Please select a valid `fem_model_type` from either `tube` or `wingbox`.")

    failure_cls = FailureExact if surface["exact_failure_constraint"] else FailureKS
    subs["failure"] = failure_cls(surface=surface, promotes_inputs=[stress_name], promotes_outputs=["failure"])
    return {"subsystems": subs}


class SpatialBeamFunctionals(om.Group):
    """Group that contains the spatial beam functionals used to evaluate performance."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_spatial_beam_functionals_kwargs(f.surface))
        return data
