from typing import Any

import jax.numpy as jnp
from pydantic import model_validator

from om4.components.jax_comps import JaxExplicitComp

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


def _build_rotation_matrix(alpha, beta):
    """Aero -> wind rotation matrix, built from scalar alpha/beta (shape (1,)) jnp arrays."""
    cosa = jnp.squeeze(jnp.cos(alpha))
    sina = jnp.squeeze(jnp.sin(alpha))
    cosb = jnp.squeeze(jnp.cos(beta))
    sinb = jnp.squeeze(jnp.sin(beta))

    row0 = jnp.stack([cosb * cosa, -sinb, cosb * sina])
    row1 = jnp.stack([sinb * cosa, cosb, sinb * sina])
    row2 = jnp.stack([-sina, jnp.zeros_like(sina), cosa])
    return jnp.stack([row0, row1, row2], axis=0)


def _rotate_last_axis(arr, Tw):
    """Apply the 3x3 rotation matrix ``Tw`` to the last axis of ``arr`` (shape (..., 3))."""
    return jnp.einsum("lk,...k->...l", Tw, arr)


class RotateToWindFrame(JaxExplicitComp):
    """
    Rotate the VLM Geometry from the standard aerodynamic to the wind frame.
    In the wind frame the freestream will be along the x-axis.

    This transformation is given by the following rotation matrix:

    | x_wind |   | cosb*cosa, -sinb, cosb*sina |   | x_aero |
    | y_wind | = | sinb*cosa,  cosb, sinb*sina | . | y_aero |
    | z_wind |   | -sina,         0,      cosa |   | z_aero |


    Where "a" is the angle of attack and "b" is the sideslip angle.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in aero
        frame.
    bound_vecs[num_eval_points, 3] : numpy array
        The vectors representing the bound vortices for each panel in the
        problem.
        This array contains points for all lifting surfaces in the problem.
    coll_pts[num_eval_points, 3] : numpy array
        The xyz coordinates of the collocation points used in the VLM analysis.
        This array contains points for all lifting surfaces in the problem.
    force_pts[num_eval_points, 3] : numpy array
        The xyz coordinates of the force points used in the VLM analysis.
        We evaluate the velocity of the air at these points to get the sectional
        forces acting on the panel. This includes both the freestream and the
        induced velocity acting at these points.
        This array contains points for all lifting surfaces in the problem.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in aero frame, computed as the cross of
        the two diagonals from the mesh points.
    rotational_velocities[num_eval_points, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces.
        This array contains points for all lifting surfaces in the problem.
    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    def_mesh_w_frame[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in wind
        frame.
    bound_vecs_w_frame[num_eval_points, 3] : numpy array
        Bound points for the horseshoe vortices in wind frame.
    coll_pts_w_frame[num_eval_points, 3] : numpy array
        Collocation points on the 3/4 chord line where the flow tangency
        condition is satisfed in wind frame.
    force_pts_w_frame[num_eval_points, 3] : numpy array
        Force pts in wind frame.
    normals_w_frame[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in wind frame.
    rotational_velocities_w_frame[num_eval_points, 3] : numpy array
        Velocity component at collecation points due to rotational velocity in
        wind frame.
    """

    surfaces: list[Surface]
    rotational: bool = False

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surfaces = f["surfaces"]
        rotational = f["rotational"]

        # Loop through all the surfaces to determine the total number
        # of evaluation points.
        num_eval_points = 0
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            num_eval_points += (nx - 1) * (ny - 1)

        d.add_input("alpha", val=0.0, units="rad", tags=["mphys_input"])
        d.add_input("beta", val=0.0, units="rad", tags=["mphys_input"])
        d.add_input("coll_pts", shape=(num_eval_points, 3), units="m")
        d.add_input("force_pts", shape=(num_eval_points, 3), units="m")
        d.add_input("bound_vecs", shape=(num_eval_points, 3), units="m")

        d.add_output("coll_pts_w_frame", shape=(num_eval_points, 3), units="m")
        d.add_output("force_pts_w_frame", shape=(num_eval_points, 3), units="m")
        d.add_output("bound_vecs_w_frame", shape=(num_eval_points, 3), units="m")

        if rotational:
            d.add_input("rotational_velocities", shape=(num_eval_points, 3), units="m/s")
            d.add_output("rotational_velocities_w_frame", shape=(num_eval_points, 3), units="m/s")

        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            d.add_input(f"{name}_def_mesh", shape=(nx, ny, 3), units="m")
            d.add_input(f"{name}_normals", shape=(nx - 1, ny - 1, 3))

            d.add_output(f"{name}_def_mesh_w_frame", shape=(nx, ny, 3), units="m")
            d.add_output(f"{name}_normals_w_frame", shape=(nx - 1, ny - 1, 3))

        # Derivatives are handled automatically by JaxExplicitComp (asdex-detected
        # sparsity + autodiff), so no declare_partials calls are needed here.
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None, statics=None):
        Tw = _build_rotation_matrix(inputs["alpha"], inputs["beta"])

        outputs["bound_vecs_w_frame"] = _rotate_last_axis(inputs["bound_vecs"], Tw)
        outputs["coll_pts_w_frame"] = _rotate_last_axis(inputs["coll_pts"], Tw)
        outputs["force_pts_w_frame"] = _rotate_last_axis(inputs["force_pts"], Tw)

        if self.rotational:
            outputs["rotational_velocities_w_frame"] = _rotate_last_axis(inputs["rotational_velocities"], Tw)

        for surface in self.surfaces:
            name = surface["name"]

            outputs[f"{name}_def_mesh_w_frame"] = _rotate_last_axis(inputs[f"{name}_def_mesh"], Tw)
            outputs[f"{name}_normals_w_frame"] = _rotate_last_axis(inputs[f"{name}_normals"], Tw)


class RotateFromWindFrame(JaxExplicitComp):
    """
    Rotate the aerodynamic sectional and nodal force vectors from the wind to
    the standard aerodynamic frame. This is the reverse operation of the
    RotateToWindFrame component.

    This transformation is given by the following rotation matrix:

    | F_x_aero |   | cosb*cosa, sinb*cosa, -sina |   | F_x_wind |
    | F_y_aero | = | -sinb,          cosb,     0 | . | F_y_wind |
    | F_z_aero |   | cosb*sina, sinb*sina,  cosa |   | F_z_wind |


    Where "a" is the angle of attack and "b" is the sideslip angle.

    Parameters
    ----------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.

    alpha : float
        Angle of attack in degrees.
    beta : float
        Sideslip angle in degrees.

    Returns
    -------
    sec_forces[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in aero frame.
    """

    surfaces: list[Surface]

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        d.add_input("alpha", val=0.0, units="rad", tags=["mphys_input"])
        d.add_input("beta", val=0.0, units="rad", tags=["mphys_input"])

        for surface in f["surfaces"]:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            d.add_input(f"{name}_sec_forces_w_frame", shape=(nx - 1, ny - 1, 3), units="N")
            d.add_output(f"{name}_sec_forces", shape=(nx - 1, ny - 1, 3), units="N", tags=["mphys_coupling"])

        # Derivatives are handled automatically by JaxExplicitComp (asdex-detected
        # sparsity + autodiff), so no declare_partials calls are needed here.
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None, statics=None):
        # wind->aero rotation matrix is the transpose of aero->wind.
        TwT = _build_rotation_matrix(inputs["alpha"], inputs["beta"]).T

        for surface in self.surfaces:
            name = surface["name"]
            outputs[f"{name}_sec_forces"] = _rotate_last_axis(inputs[f"{name}_sec_forces_w_frame"], TwT)
