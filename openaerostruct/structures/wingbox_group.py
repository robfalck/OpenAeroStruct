from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.section_properties_wingbox import SectionPropertiesWingbox
from openaerostruct.structures.tube_group import _span_spline
from openaerostruct.structures.wingbox_geometry import WingboxGeometry
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _wingbox_group_kwargs(surface: Surface) -> dict:
    subs = {}
    input_defaults = {}

    for cp_key, name in (("spar_thickness_cp", "spar_thickness"), ("skin_thickness_cp", "skin_thickness")):
        if cp_key in surface:
            subs[f"{name}_bsp"], input_defaults[cp_key] = _span_spline(surface, cp_key, name, "m")

    subs["wingbox_geometry"] = WingboxGeometry(
        surface=surface,
        promotes_inputs=["mesh"],
        promotes_outputs=["fem_chords", "fem_twists", "streamwise_chords"],
    )

    subs["wingbox"] = SectionPropertiesWingbox(
        surface=surface,
        promotes_inputs=[
            "spar_thickness",
            "skin_thickness",
            "t_over_c",
            "fem_chords",
            "fem_twists",
            "streamwise_chords",
        ],
        promotes_outputs=["A", "Iy", "Qz", "Iz", "J", "A_enc", "A_int", "htop", "hbottom", "hfront", "hrear"],
    )
    return {"subsystems": subs, "input_defaults": input_defaults}


class WingboxGroup(om.Group):
    """Group that contains everything needed for a structural-only problem."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, om.InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_wingbox_group_kwargs(f.surface))
        return data
