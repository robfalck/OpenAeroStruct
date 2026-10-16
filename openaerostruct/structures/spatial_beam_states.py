from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.compute_point_mass_loads import ComputePointMassLoads
from openaerostruct.structures.compute_thrust_loads import ComputeThrustLoads
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.disp import Disp
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.structures.total_loads import TotalLoads
from openaerostruct.structures.wing_weight_loads import StructureWeightLoads
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _spatial_beam_states_kwargs(surface: Surface) -> dict:
    subs = {}
    promotes = []

    if surface["struct_weight_relief"]:
        subs["struct_weight_loads"] = StructureWeightLoads(
            surface=surface,
            promotes_inputs=["element_mass", "nodes", "load_factor"],
            promotes_outputs=["struct_weight_loads"],
        )
        promotes.append("struct_weight_loads")

    if surface["distributed_fuel_weight"]:
        subs["fuel_loads"] = FuelLoads(
            surface=surface,
            promotes_inputs=["nodes", "load_factor", "fuel_vols", "fuel_mass"],
            promotes_outputs=["fuel_weight_loads"],
        )
        promotes.append("fuel_weight_loads")

    if "n_point_masses" in surface:
        subs["point_masses"] = ComputePointMassLoads(
            surface=surface,
            promotes_inputs=["point_mass_locations", "point_masses", "nodes", "load_factor"],
            promotes_outputs=["loads_from_point_masses"],
        )
        promotes.append("loads_from_point_masses")

        subs["thrust_loads"] = ComputeThrustLoads(
            surface=surface,
            promotes_inputs=["point_mass_locations", "engine_thrusts", "nodes"],
            promotes_outputs=["loads_from_thrusts"],
        )
        promotes.append("loads_from_thrusts")

    subs["total_loads"] = TotalLoads(
        surface=surface, promotes_inputs=["loads"] + promotes, promotes_outputs=["total_loads"]
    )
    subs["create_rhs"] = CreateRHS(surface=surface, promotes_inputs=["total_loads"], promotes_outputs=["forces"])
    subs["fem"] = FEM(surface=surface, promotes_inputs=["*"], promotes_outputs=["*"])
    subs["disp"] = Disp(surface=surface, promotes_inputs=["*"], promotes_outputs=["*"])
    return {"subsystems": subs}


class SpatialBeamStates(om.Group):
    """Group that contains the spatial beam states."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_spatial_beam_states_kwargs(f.surface))
        return data
