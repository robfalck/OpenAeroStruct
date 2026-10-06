from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.aerodynamics.lift_drag import LiftDrag
from openaerostruct.aerodynamics.coeffs import Coeffs
from openaerostruct.aerodynamics.total_lift import TotalLift
from openaerostruct.aerodynamics.total_drag import TotalDrag
from openaerostruct.aerodynamics.viscous_drag import ViscousDrag
from openaerostruct.aerodynamics.wave_drag import WaveDrag
from openaerostruct.aerodynamics.lift_coeff_2D import LiftCoeff2D
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _vlm_functionals_kwargs(surface: Surface) -> dict:
    """Return the Group spec for VLMFunctionals."""
    sub = om.Subsystem
    return {
        "subsystems": {
            "liftcoeff": sub(
                LiftCoeff2D(surface=surface),
                promotes_inputs=["v", "alpha", "rho", "widths", "chords", "sec_forces"],
                promotes_outputs=["Cl"],
            ),
            "liftdrag": sub(
                LiftDrag(surface=surface),
                promotes_inputs=["alpha", "beta", "sec_forces"],
                promotes_outputs=["L", "D"],
            ),
            "coeffs": sub(Coeffs(), promotes_inputs=["v", "rho", "S_ref", "L", "D"], promotes_outputs=["CL1", "CDi"]),
            "CL": sub(TotalLift(surface=surface), promotes_inputs=["CL1"], promotes_outputs=["CL"]),
            "viscousdrag": sub(
                ViscousDrag(surface=surface),
                promotes_inputs=["Mach_number", "re", "widths", "lengths_spanwise", "lengths", "S_ref", "t_over_c"],
                promotes_outputs=["CDv"],
            ),
            "wavedrag": sub(
                WaveDrag(surface=surface),
                promotes_inputs=["Mach_number", "lengths_spanwise", "widths", "CL", "chords", "t_over_c"],
                promotes_outputs=["CDw"],
            ),
            "CD": sub(TotalDrag(surface=surface), promotes_inputs=["CDv", "CDi", "CDw"], promotes_outputs=["CD"]),
        }
    }


class VLMFunctionals(om.Group):
    """
    Group that contains the aerodynamic functionals used to evaluate
    performance. These are not included in the coupled aerostructural group,
    but are only used to compute aerodynamic performance. This includes
    computing lift, drag, CL, CD, viscous CD, and wave drag CD.
    """

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, om.Subsystem | Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data.update(_vlm_functionals_kwargs(field_values(cls, data).surface))
        return data
