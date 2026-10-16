from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.geometry.radius_comp import RadiusComp
from openaerostruct.structures.section_properties_tube import SectionPropertiesTube
from openaerostruct.utils.bspline_comp import BsplineComp
from openaerostruct.utils.interpolation import get_normalized_span_coords
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _span_spline(surface: Surface, cp_key: str, name: str, units: str) -> tuple[BsplineComp, om.InputDefault]:
    n_cp = len(surface[cp_key])
    cp_val = np.asarray(surface[cp_key], dtype=float)
    comp = BsplineComp(
        num_cp=n_cp,
        x_interp=get_normalized_span_coords(surface, mid_panel=True),
        order=min(n_cp, 4),
        x_cp_start=0.0,
        x_cp_end=1.0,
        cp_name=cp_key,
        interp_name=name,
        units=units,
        cp_val=cp_val,
        promotes_inputs=[cp_key],
        promotes_outputs=[name],
    )
    return comp, om.InputDefault(val=cp_val, units=units)


def _tube_group_kwargs(surface: Surface) -> dict:
    subs = {}
    input_defaults = {}

    for cp_key, name in (("thickness_cp", "thickness"), ("radius_cp", "radius")):
        if cp_key in surface:
            subs[f"{name}_bsp"], input_defaults[cp_key] = _span_spline(surface, cp_key, name, "m")

    if "radius_cp" not in surface:
        subs["radius_comp"] = RadiusComp(
            surface=surface, promotes_inputs=["mesh", "t_over_c"], promotes_outputs=["radius"]
        )

    subs["tube"] = SectionPropertiesTube(
        surface=surface,
        promotes_inputs=["thickness", "radius"],
        promotes_outputs=["A", "Iy", "Iz", "J"],
    )
    return {"subsystems": subs, "input_defaults": input_defaults}


class TubeGroup(om.Group):
    """Group that contains everything needed for a structural-only problem."""

    surface: Surface
    connect_geom_DVs: bool = Field(default=True, description="No longer necessary; kept for backward compatibility.")

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, om.InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_tube_group_kwargs(f.surface))
        return data
