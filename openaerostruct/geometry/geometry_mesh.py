"""Group that manipulates geometry mesh based on high-level design parameters."""

from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.geometry.geometry_mesh_transformations import (
    Taper,
    ScaleX,
    Sweep,
    ShearX,
    Stretch,
    ShearY,
    Dihedral,
    ShearZ,
    Rotate,
)
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


_CHAIN = ["taper", "scale_x", "sweep", "shear_x", "stretch", "shear_y", "dihedral", "shear_z", "rotate"]


def _geometry_mesh_kwargs(surface: Surface) -> dict:
    """
    Return the Group spec for GeometryMesh: the chain of mesh transformations for one surface.

    A transformation's driving input is promoted only when the surface defines the matching
    parameter, so that only active parameters appear on the parent.
    """
    ref_axis_pos = surface["ref_axis_pos"] if "ref_axis_pos" in surface else 0.25

    mesh = np.asarray(surface["mesh"], dtype=float)
    ny = mesh.shape[1]
    mesh_shape = mesh.shape
    symmetry = surface["symmetry"]

    def prom(key, name):
        return [name] if key in surface else []

    subs = {}

    # 1. Taper
    subs["taper"] = om.Subsystem(
        Taper(
            val=surface["taper"] if "taper" in surface else 1.0,
            mesh=mesh,
            symmetry=symmetry,
            ref_axis_pos=ref_axis_pos,
        ),
        promotes_inputs=prom("taper", "taper"),
    )

    # 2. Scale X
    subs["scale_x"] = om.Subsystem(
        ScaleX(val=np.ones(ny), mesh_shape=mesh_shape, ref_axis_pos=ref_axis_pos),
        promotes_inputs=prom("chord_cp", "chord"),
    )

    # 3. Sweep
    subs["sweep"] = om.Subsystem(
        Sweep(val=surface["sweep"] if "sweep" in surface else 0.0, mesh_shape=mesh_shape, symmetry=symmetry),
        promotes_inputs=prom("sweep", "sweep"),
    )

    # 4. Shear X
    subs["shear_x"] = om.Subsystem(
        ShearX(val=np.zeros(ny), mesh_shape=mesh_shape), promotes_inputs=prom("xshear_cp", "xshear")
    )

    # 5. Stretch
    if "span" in surface:
        span = surface["span"]
    else:
        # Compute span. We need .real to make span to avoid OpenMDAO warnings.
        ref_axis = ref_axis_pos * mesh[-1, :, :] + (1 - ref_axis_pos) * mesh[0, :, :]
        span = max(ref_axis[:, 1]).real - min(ref_axis[:, 1]).real
        if symmetry:
            span *= 2.0
    subs["stretch"] = om.Subsystem(
        Stretch(val=float(span), mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
        promotes_inputs=prom("span", "span"),
    )

    # 6. Shear Y
    subs["shear_y"] = om.Subsystem(
        ShearY(val=np.zeros(ny), mesh_shape=mesh_shape), promotes_inputs=prom("yshear_cp", "yshear")
    )

    # 7. Dihedral
    subs["dihedral"] = om.Subsystem(
        Dihedral(val=surface["dihedral"] if "dihedral" in surface else 0.0, mesh_shape=mesh_shape, symmetry=symmetry),
        promotes_inputs=prom("dihedral", "dihedral"),
    )

    # 8. Shear Z
    subs["shear_z"] = om.Subsystem(
        ShearZ(val=np.zeros(ny), mesh_shape=mesh_shape), promotes_inputs=prom("zshear_cp", "zshear")
    )

    # 9. Rotate
    subs["rotate"] = om.Subsystem(
        Rotate(val=np.zeros(ny), mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
        promotes_inputs=prom("twist_cp", "twist"),
        promotes_outputs=["mesh"],
    )

    connections = [om.Connection(src=f"{a}.mesh", tgt=f"{b}.in_mesh") for a, b in zip(_CHAIN[:-1], _CHAIN[1:])]

    return {"subsystems": subs, "connections": connections}


class GeometryMesh(om.Group):
    """
    Group that performs mesh manipulation functions.

    It reads the initial mesh from the surface and outputs the altered mesh based on the
    geometric design variables.  Only parameters the surface defines are promoted (and so
    can deform the mesh from above); the rest stay at their identity values.

    Parameters
    ----------
    sweep : float
        Shearing sweep angle in degrees.
    dihedral : float
        Dihedral angle in degrees.
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.
    chord[ny] : numpy array
        Spanwise distribution of the chord scaler.
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface and the geometric design variables.
    """

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, om.Subsystem | Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[om.Connection] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_geometry_mesh_kwargs(f.surface))
        return data
