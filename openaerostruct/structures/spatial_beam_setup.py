from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.assemble_k_group import AssembleKGroup
from openaerostruct.structures.compute_nodes import ComputeNodes
from openaerostruct.structures.fuel_vol import WingboxFuelVol
from openaerostruct.structures.structural_cg import StructuralCG
from openaerostruct.structures.weight import Weight
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _spatial_beam_setup_kwargs(surface: Surface) -> dict:
    subs = {
        "nodes": ComputeNodes(surface=surface, promotes_inputs=["mesh"], promotes_outputs=["nodes"]),
        "assembly": AssembleKGroup(
            surface=surface,
            promotes_inputs=["A", "Iy", "Iz", "J", "nodes"],
            promotes_outputs=["local_stiff_transformed"],
        ),
        "structural_mass": Weight(
            surface=surface,
            promotes_inputs=["A", "nodes"],
            promotes_outputs=["structural_mass", "element_mass"],
        ),
        "structural_cg": StructuralCG(
            surface=surface,
            promotes_inputs=["nodes", "structural_mass", "element_mass"],
            promotes_outputs=["cg_location"],
        ),
    }
    if "wingbox" in surface["fem_model_type"].lower():
        subs["fuel_vol"] = WingboxFuelVol(
            surface=surface, promotes_inputs=["nodes", "A_int"], promotes_outputs=["fuel_vols"]
        )
    return {"subsystems": subs}


class SpatialBeamSetup(om.Group):
    """Group that sets up the spatial beam components and assembles the stiffness matrix."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_spatial_beam_setup_kwargs(f.surface))
        return data
