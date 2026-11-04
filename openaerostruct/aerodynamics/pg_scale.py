from typing import Any

import jax.numpy as jnp
import numpy as np
from pydantic import model_validator

from om4.components.jax_comps import JaxExplicitComp

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


def _scale_last_axis(arr, factors):
    """Multiply each of the 3 components along ``arr``'s last axis by its own factor."""
    return jnp.stack([arr[..., i] * factors[i] for i in range(3)], axis=-1)


class ScaleToPrandtlGlauert(JaxExplicitComp):
    """
    Scale the wind frame coordinates to get the Prandtl-Glauert transformed
    geometry.

    The Prandtl glauert transformation is defined as below:
        Coordinates
        x_pg = x_wind
        y_pg = B*y_wind
        z_pg = B*z_wind

        Normals
        n_x_pg = B*n_x_wind
        n_y_pg = n_y_wind
        n_z_pg = n_z_wind

        Perturbation velocities
        v_x_pg = B^2*v_x_wind
        v_y_pg = B*v_y_wind
        v_z_pg = B*v_z_wind

    where B = sqrt(1 - M^2).
    Note: The freestream velocity remains untransformed and therefore is not
    included in this component.

    Parameters
    ----------
    def_mesh_w_frame[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in aero
        frame.
    bound_vecs_w_frame[num_eval_points, 3] : numpy array
        The vectors representing the bound vortices for each panel in the
        problem.
        This array contains points for all lifting surfaces in the problem.
    coll_pts_w_frame[num_eval_points, 3] : numpy array
        The xyz coordinates of the collocation points used in the VLM analysis.
        This array contains points for all lifting surfaces in the problem.
    force_pts_w_frame[num_eval_points, 3] : numpy array
        The xyz coordinates of the force points used in the VLM analysis.
        We evaluate the velocity of the air at these points to get the sectional
        forces acting on the panel. This includes both the freestream and the
        induced velocity acting at these points.
        This array contains points for all lifting surfaces in the problem.
    normals_w_frame[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in aero frame, computed as the cross of
        the two diagonals from the mesh points.
    rotational_velocities_w_frame[num_eval_points, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces.
        This array contains points for all lifting surfaces in the problem.

    M : float
        Freestream Mach number.

    Returns
    -------
    def_mesh_pg[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface in PG frame.
    bound_vecs_pg[num_eval_points, 3] : numpy array
        Bound points in PG frame.
    coll_pts_pg[num_eval_points, 3] : numpy array
        Collocation points in PG frame.
    force_pts_pg[num_eval_points, 3] : numpy array
        Force points in PG frame.
    normals_pg[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel in PG frame.
    rotational_velocities_pg[num_eval_points, 3] : numpy array
        Velocity component at collocation points due to rotational velocity in PG frame.
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

        d.add_input("Mach_number", val=0.0, tags=["mphys_input"])

        d.add_input("coll_pts_w_frame", shape=(num_eval_points, 3), units="m")
        d.add_input("force_pts_w_frame", shape=(num_eval_points, 3), units="m")
        d.add_input("bound_vecs_w_frame", shape=(num_eval_points, 3), units="m")

        d.add_output("coll_pts_pg", shape=(num_eval_points, 3), units="m")
        d.add_output("force_pts_pg", shape=(num_eval_points, 3), units="m")
        d.add_output("bound_vecs_pg", shape=(num_eval_points, 3), units="m")

        if rotational:
            d.add_input("rotational_velocities_w_frame", shape=(num_eval_points, 3), units="m/s")
            d.add_output("rotational_velocities_pg", shape=(num_eval_points, 3), units="m/s")

        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            d.add_input(f"{name}_def_mesh_w_frame", shape=(nx, ny, 3), units="m")
            d.add_input(f"{name}_normals_w_frame", shape=(nx - 1, ny - 1, 3))

            d.add_output(f"{name}_def_mesh_pg", shape=(nx, ny, 3), units="m")
            d.add_output(f"{name}_normals_pg", shape=(nx - 1, ny - 1, 3))

        # Derivatives are handled automatically by JaxExplicitComp (asdex-detected
        # sparsity + autodiff), so no declare_partials calls are needed here.
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None, statics=None):
        M = inputs["Mach_number"]
        betaPG = jnp.sqrt(1 - M**2)

        fact = (1.0, betaPG, betaPG)
        outputs["bound_vecs_pg"] = _scale_last_axis(inputs["bound_vecs_w_frame"], fact)
        outputs["coll_pts_pg"] = _scale_last_axis(inputs["coll_pts_w_frame"], fact)
        outputs["force_pts_pg"] = _scale_last_axis(inputs["force_pts_w_frame"], fact)

        if self.rotational:
            fact_rot = (betaPG**2, betaPG, betaPG)
            outputs["rotational_velocities_pg"] = _scale_last_axis(
                inputs["rotational_velocities_w_frame"], fact_rot
            )

        fact_norm = (betaPG, 1.0, 1.0)
        for surface in self.surfaces:
            name = surface["name"]

            outputs[f"{name}_def_mesh_pg"] = _scale_last_axis(inputs[f"{name}_def_mesh_w_frame"], fact)
            outputs[f"{name}_normals_pg"] = _scale_last_axis(inputs[f"{name}_normals_w_frame"], fact_norm)


class ScaleFromPrandtlGlauert(JaxExplicitComp):
    """
    Scale the Prandtl-Glauert transformed forces to get the physical forces
    Prandtl-Glauert transformed geometry.

    The inverse Prandtl-Glauert transformation for forces is defined as below:
        F_x_wind = F_x_pg/B^4
        F_y_wind = F_y_pg/B^3
        F_z_wind = F_z_pg/B^3

    where B = sqrt(1 - M^2).

    Parameters
    ----------
    sec_forces_pg[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in PG domain.

    M : float
        Freestream Mach number.

    Returns
    -------
    sec_forces_w_frame[nx-1, ny-1, 3] : numpy array
        Force vectors on each panel (lattice) in wind frame.
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

        d.add_input("Mach_number", val=0.0, tags=["mphys_input"])

        for surface in f["surfaces"]:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            d.add_input(f"{name}_sec_forces_pg", val=np.zeros((nx - 1, ny - 1, 3)), units="N")
            d.add_output(f"{name}_sec_forces_w_frame", val=np.zeros((nx - 1, ny - 1, 3)), units="N")

        # Derivatives are handled automatically by JaxExplicitComp (asdex-detected
        # sparsity + autodiff), so no declare_partials calls are needed here.
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None, statics=None):
        M = inputs["Mach_number"]
        betaPG = jnp.sqrt(1 - M**2)
        fact = (1.0 / betaPG**4, 1.0 / betaPG**3, 1.0 / betaPG**3)

        for surface in self.surfaces:
            name = surface["name"]
            outputs[f"{name}_sec_forces_w_frame"] = _scale_last_axis(inputs[f"{name}_sec_forces_pg"], fact)
