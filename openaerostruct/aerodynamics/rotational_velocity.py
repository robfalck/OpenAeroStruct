from functools import cached_property
from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface



class RotationalVelocity(om.ExplicitComponent):
    """
    Compute the velocity due to rigid body rotation.

    Parameters
    ----------
    omega[3] : ndarray
        Angular velocity vector for each surface about center of gravity.
        Only available if the rotational options is set to True.
    cg[3] : ndarray
        The x, y, z coordinates of the center of gravity for the entire aircraft.
        Only available if the rotational options is set to True.
    coll_pts[num_eval_points, 3] : ndarray
        The xyz coordinates of the collocation points used in the VLM analysis.
        This array contains points for all lifting surfaces in the problem.
        Only available if the rotational options is set to True.

    Returns
    -------
    rotational_velocities[num_eval_points, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces.
        This array contains points for all lifting surfaces in the problem.
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

        surfaces = f["surfaces"]

        system_size = 0
        sizes = []

        # Loop through each surface and cumulatively add the number of panels
        # to obtain system_size.
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            size = (nx - 1) * (ny - 1)
            system_size += size
            sizes.append(size)

        d.add_input("coll_pts", shape=(system_size, 3), units="m")
        d.add_input("omega", val=np.zeros((3,)), units="rad/s", tags=["mphys_input"])
        d.add_input("cg", val=np.ones((3,)), units="m", tags=["mphys_input"])

        d.add_output("rotational_velocities", shape=(system_size, 3), units="m/s")

        # First Half of cross product
        row = np.array([1, 2, 0])
        col = np.array([2, 0, 1])

        rows1 = np.tile(row, system_size) + np.repeat(3 * np.arange(system_size), 3)
        cols1 = np.tile(col, system_size)

        # Second Half of cross product
        rows2 = np.tile(col, system_size) + np.repeat(3 * np.arange(system_size), 3)
        cols2 = np.tile(row, system_size)

        rows = np.concatenate([rows1, rows2])
        cols = np.concatenate([cols1, cols2])

        d.declare_partials("rotational_velocities", "cg", rows=rows, cols=cols)
        d.declare_partials("rotational_velocities", "omega", rows=rows, cols=cols)

        cols1 = np.tile(col, system_size) + np.repeat(3 * np.arange(system_size), 3)
        cols2 = np.tile(row, system_size) + np.repeat(3 * np.arange(system_size), 3)
        cols = np.concatenate([cols1, cols2])

        d.declare_partials("rotational_velocities", "coll_pts", rows=rows, cols=cols)
        return d.into(_spec)

    @cached_property
    def system_size(self) -> int:
        """Total number of VLM panels over all surfaces."""
        return vlm_system_size(self.surfaces)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Angular velocity term
        cg = inputs["cg"]
        omega = inputs["omega"]
        c_pts = inputs["coll_pts"]

        for j in np.arange(c_pts.shape[0]):
            r = c_pts[j, :] - cg

            outputs["rotational_velocities"][j, :] = np.cross(omega, r)

    def compute_partials(self, inputs, J):
        cg = inputs["cg"]
        omega = inputs["omega"]
        c_pts = inputs["coll_pts"]

        surfaces = self.surfaces
        idx = jdx = 0
        ii = self.system_size * 3
        # om4 subjacs are write-only (ai/OM4_NEEDS.md N-005): fill locally, assign once
        d_cg = np.zeros(2 * ii)
        d_cpts = np.zeros(2 * ii)
        d_omega = np.zeros(2 * ii)
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            size = (nx - 1) * (ny - 1)

            r = c_pts[jdx : jdx + size, :] - cg

            # Cross product derivatives organized so we can tile a variable directly into slices

            d_cg[idx : idx + size * 3] = np.tile(omega, size)
            d_cg[idx + ii : idx + ii + size * 3] = -np.tile(omega, size)

            d_cpts[idx : idx + size * 3] = -np.tile(omega, size)
            d_cpts[idx + ii : idx + ii + size * 3] = np.tile(omega, size)

            d_omega[idx : idx + size * 3] = r.flatten()
            d_omega[idx + ii : idx + ii + size * 3] = -r.flatten()

            idx += 3 * size
            jdx += size

        J["rotational_velocities", "cg"] = d_cg
        J["rotational_velocities", "coll_pts"] = d_cpts
        J["rotational_velocities", "omega"] = d_omega
