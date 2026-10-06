from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.functionals.sum_areas import SumAreas
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _total_aero_performance_kwargs(surfaces: list[Surface], user_specified_Sref: bool) -> dict:
    """Return the Group spec for TotalAeroPerformance."""
    subs = {}
    if not user_specified_Sref:
        subs["sum_areas"] = om.Subsystem(
            SumAreas(surfaces=surfaces), promotes_inputs=["*S_ref"], promotes_outputs=["S_ref_total"]
        )

    subs["CL_CD"] = om.Subsystem(
        TotalLiftDrag(surfaces=surfaces),
        promotes_inputs=["*CL", "*CD", "*S_ref", "S_ref_total", "rho", "v"],
        promotes_outputs=["CL", "CD", "L", "D"],
    )

    subs["moment"] = om.Subsystem(
        MomentCoefficient(surfaces=surfaces),
        promotes_inputs=["v", "cg", "rho", "*S_ref", "*b_pts", "*widths", "*chords", "*sec_forces", "S_ref_total"],
        promotes_outputs=["CM"],
    )
    return {"subsystems": subs}


class TotalAeroPerformance(om.Group):
    """
    Group to contain the total aerodynamic performance components.
    """

    surfaces: list[Surface]
    user_specified_Sref: bool

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, om.Subsystem | Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_total_aero_performance_kwargs(f.surfaces, f.user_specified_Sref))
        return data
