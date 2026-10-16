from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.functionals.sum_areas import SumAreas
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _total_performance_kwargs(
    surfaces: list[Surface], user_specified_Sref: bool, internally_connect_fuelburn: bool
) -> dict:
    """Return the Group spec for TotalPerformance."""
    subs = {}
    if not user_specified_Sref:
        subs["sum_areas"] = SumAreas(surfaces=surfaces, promotes_inputs=["*S_ref"], promotes_outputs=["S_ref_total"])

    promote_fuelburn = ["fuelburn"] if internally_connect_fuelburn else []

    subs["CL_CD"] = TotalLiftDrag(
        surfaces=surfaces,
        promotes_inputs=["*CL", "*CD", "*S_ref", "S_ref_total", "rho", "v"],
        promotes_outputs=["CL", "CD", "L", "D"],
    )

    subs["fuelburn"] = BreguetRange(
        surfaces=surfaces,
        promotes_inputs=["*structural_mass", "CL", "CD", "CT", "speed_of_sound", "R", "Mach_number", "W0"],
        promotes_outputs=["fuelburn"],
    )

    subs["L_equals_W"] = Equilibrium(
        surfaces=surfaces,
        promotes_inputs=["CL", "*structural_mass", "S_ref_total", "W0", "load_factor", "rho", "v"] + promote_fuelburn,
        promotes_outputs=["L_equals_W", "total_weight"],
    )

    subs["CG"] = CenterOfGravity(
        surfaces=surfaces,
        promotes_inputs=["*structural_mass", "*cg_location", "total_weight", "W0", "empty_cg", "load_factor"]
        + promote_fuelburn,
        promotes_outputs=["cg"],
    )

    subs["moment"] = MomentCoefficient(
        surfaces=surfaces,
        promotes_inputs=["v", "rho", "cg", "S_ref_total", "*b_pts", "*widths", "*chords", "*sec_forces", "*S_ref"],
        promotes_outputs=["CM"],
    )

    # The leaves disagree on the defaults of these promoted inputs, which om4 rejects at setup
    input_defaults = {
        "W0": om.InputDefault(val=np.ones(1), units="kg"),
        "v": om.InputDefault(val=np.ones(1), units="m/s"),
        "rho": om.InputDefault(val=np.ones(1), units="kg/m**3"),
    }
    if user_specified_Sref:
        input_defaults["S_ref_total"] = om.InputDefault(val=np.ones(1), units="m**2")
    return {"subsystems": subs, "input_defaults": input_defaults}


class TotalPerformance(om.Group):
    """
    Group to contain the total aerostructural performance components.
    """

    surfaces: list[Surface]
    user_specified_Sref: bool
    internally_connect_fuelburn: bool = True

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, om.InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_total_performance_kwargs(f.surfaces, f.user_specified_Sref, f.internally_connect_fuelburn))
        return data
