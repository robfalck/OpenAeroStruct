from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.geometry.geometry_mesh import GeometryMesh
from openaerostruct.utils.bspline_comp import BsplineComp
from openaerostruct.utils.interpolation import get_normalized_span_coords
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


# (control-point key, interpolated name, units, interpolate at panel midpoints, x_cp range)
_SPLINES = [
    ("twist_cp", "twist", "deg", False, None),
    ("chord_cp", "chord", None, False, None),
    ("t_over_c_cp", "t_over_c", None, True, (0.0, 1.0)),
    ("xshear_cp", "xshear", "m", False, None),
    ("yshear_cp", "yshear", "m", False, None),
    ("zshear_cp", "zshear", "m", False, None),
]

# Scalar geometric parameters passed straight to GeometryMesh: (key, units)
_SCALARS = [("sweep", "deg"), ("span", "m"), ("dihedral", "deg"), ("taper", None)]


def _resolve_gen_mesh(surface: Surface) -> Surface:
    """Build the mesh from the section keys when the surface asks for 'gen-mesh'."""
    if not (isinstance(surface.mesh, str) and surface.mesh == "gen-mesh"):
        return surface
    from openaerostruct.meshing.section_mesh_generator import generate_mesh

    mesh, _ = generate_mesh(surface.model_copy(update={"num_sections": 1}))
    # Reset taper and sweep so that OAS doesn't apply the transformations again
    return surface.model_copy(update={"num_sections": 1, "mesh": mesh, "taper": 1.0, "sweep": 0.0})


def _geometry_kwargs(surface: Surface) -> dict:
    """
    Return the Group spec for Geometry: B-splines for each active control-point set plus GeometryMesh.

    A parameter is active when the surface defines it.  Its control points (or scalar value)
    get an input default unless the surface sets ``<key>_dv=False``, in which case they keep
    the component default, as in OM3.
    """
    subs = {}
    input_defaults = {}
    mesh_inputs = []

    for cp_key, name, units, mid_panel, x_cp_range in _SPLINES:
        if cp_key not in surface:
            continue
        n_cp = len(surface[cp_key])
        x_cp_start, x_cp_end = x_cp_range if x_cp_range else (None, None)
        is_dv = surface.get(f"{cp_key}_dv", True)
        # om4 ignores InputDefault.val (ai/OM4_NEEDS.md B-005), so the component default must
        # carry the value too. Without the default, OM3 fell back to the spline's ones.
        cp_val = np.asarray(surface[cp_key], dtype=float) if is_dv else None
        subs[f"{name}_bsp"] = om.Subsystem(
            BsplineComp(
                num_cp=n_cp,
                x_interp=get_normalized_span_coords(surface, mid_panel=mid_panel),
                order=min(n_cp, 4),
                x_cp_start=x_cp_start,
                x_cp_end=x_cp_end,
                cp_name=cp_key,
                interp_name=name,
                units=units,
                cp_val=cp_val,
            ),
            promotes_inputs=[cp_key],
            promotes_outputs=[name],
        )
        if name != "t_over_c":
            mesh_inputs.append(name)
        if is_dv:
            input_defaults[cp_key] = om.InputDefault(val=cp_val, units=units)

    for key, units in _SCALARS:
        if key not in surface:
            continue
        mesh_inputs.append(key)
        if surface.get(f"{key}_dv", True):
            input_defaults[key] = om.InputDefault(val=np.atleast_1d(float(surface[key])), units=units)

    subs["mesh"] = om.Subsystem(
        GeometryMesh(surface=surface), promotes_inputs=mesh_inputs, promotes_outputs=["mesh"]
    )

    return {"subsystems": subs, "input_defaults": input_defaults}


class Geometry(om.Group):
    """
    Group that contains all components needed for any type of OAS problem.

    Because we use this general group, there's some logic to figure out which
    components to add and which connections to make.
    This is especially true for all of the geometric manipulation types, such
    as twist, sweep, etc., in that we handle the creation of these parameters
    differently if the user wants to have them vary in the optimization problem.

    FFD (``DVGeo``) geometry is not yet ported to om4; see ai/TECH_DEBT.md.
    """

    surface: Surface
    connect_geom_DVs: bool = Field(
        default=True, description="No longer necessary; kept for backward compatibility."
    )

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, om.Subsystem | Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, om.InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("DVGeo") is not None:
            raise NotImplementedError("FFD geometry (DVGeo) is not yet ported to om4; see ai/TECH_DEBT.md.")
        data.pop("DVGeo", None)
        f = field_values(cls, data)
        surface = _resolve_gen_mesh(f.surface)
        data["surface"] = surface
        data.update(_geometry_kwargs(surface))
        return data
