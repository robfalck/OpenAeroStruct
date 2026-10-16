from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.structures.tube_group import TubeGroup
from openaerostruct.structures.wingbox_group import WingboxGroup
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _spatial_beam_alone_kwargs(surface: Surface) -> dict:
    subs = {"geometry": Geometry(surface=surface, promotes_inputs=[], promotes_outputs=["mesh", "t_over_c"])}

    fem_model_type = surface["fem_model_type"].lower()
    if fem_model_type == "tube":
        tube_in = []
        if "thickness_cp" in surface:
            tube_in.append("thickness_cp")
        if "radius_cp" not in surface:
            tube_in += ["mesh", "t_over_c"]
        subs["tube_group"] = TubeGroup(
            surface=surface,
            promotes_inputs=tube_in,
            promotes_outputs=["A", "Iy", "Iz", "J", "radius", "thickness"],
        )
    elif fem_model_type == "wingbox":
        # Connections and nomenclature are the same for the isotropic and composite wingbox
        wingbox_in = ["mesh", "t_over_c"]
        wingbox_out = ["A", "Iy", "Iz", "J", "Qz", "A_enc", "A_int", "htop", "hbottom", "hfront", "hrear"]
        has_skin = "skin_thickness_cp" in surface
        has_spar = "spar_thickness_cp" in surface
        if has_skin and has_spar:
            wingbox_in += ["skin_thickness_cp", "spar_thickness_cp"]
            wingbox_out += ["skin_thickness", "spar_thickness"]
        elif has_skin or has_spar:
            raise NameError("Please have both skin and spar thickness as design variables, not one or the other.")
        subs["wingbox_group"] = WingboxGroup(surface=surface, promotes_inputs=wingbox_in, promotes_outputs=wingbox_out)
    else:
        raise NameError("Please select a valid `fem_model_type` from either `tube` or `wingbox`.")

    setup_in = ["mesh", "A", "Iy", "Iz", "J"]
    if fem_model_type == "wingbox":
        setup_in.append("A_int")
    subs["struct_setup"] = SpatialBeamSetup(
        surface=surface,
        promotes_inputs=setup_in,
        promotes_outputs=["nodes", "local_stiff_transformed", "structural_mass", "cg_location", "element_mass"],
    )

    promotes = []
    if surface["struct_weight_relief"]:
        promotes += ["nodes", "element_mass", "load_factor"]
    if surface["distributed_fuel_weight"]:
        promotes += ["nodes", "load_factor"]
    if "n_point_masses" in surface:
        promotes += ["point_mass_locations", "point_masses", "nodes", "load_factor", "engine_thrusts"]
    subs["struct_states"] = SpatialBeamStates(
        surface=surface,
        promotes_inputs=["local_stiff_transformed", "forces", "loads"] + list(dict.fromkeys(promotes)),
        promotes_outputs=["disp"],
    )

    if fem_model_type == "tube":
        subs["struct_funcs"] = SpatialBeamFunctionals(
            surface=surface,
            promotes_inputs=["thickness", "radius", "nodes", "disp"],
            promotes_outputs=["thickness_intersects", "vonmises", "failure"],
        )
    else:
        stress_name = "tsaiwu_sr" if surface.get("useComposite", False) else "vonmises"
        subs["struct_funcs"] = SpatialBeamFunctionals(
            surface=surface,
            promotes_inputs=[
                "spar_thickness",
                "disp",
                "Qz",
                "J",
                "A_enc",
                "htop",
                "hbottom",
                "hfront",
                "hrear",
                "nodes",
            ],
            promotes_outputs=[stress_name, "failure"],
        )
    return {"subsystems": subs}


class SpatialBeamAlone(om.Group):
    """Group that contains everything needed for a structural-only problem."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_spatial_beam_alone_kwargs(f.surface))
        return data
