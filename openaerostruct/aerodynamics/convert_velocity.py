from functools import cached_property
from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface



class ConvertVelocity(om.ExplicitComponent):
    """
    Convert the freestream velocity magnitude into a velocity vector at each
    evaluation point. In this case, each of the panels sees the same velocity.
    This really just helps us set up the velocities for use in the VLM analysis.

    Parameters
    ----------
    alpha : float
        The angle of attack for the aircraft (all lifting surfaces) in degrees.
    beta : float
        The sideslip angle for the aircraft (all lifting surfaces) in degrees.
    v : float
        The freestream velocity magnitude.
    rotational_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.

    Returns
    -------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    """

    surfaces: list[Surface]
    rotational: bool = False  # Turn on support for computing angular velocities

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

        d.add_input("alpha", val=0.0, units="deg", tags=["mphys_input"])
        d.add_input("beta", val=0.0, units="deg", tags=["mphys_input"])
        d.add_input("v", val=1.0, units="m/s", tags=["mphys_input"])

        if rotational:
            d.add_input("rotational_velocities", shape=(system_size, 3), units="m/s")

        d.add_output("freestream_velocities", shape=(system_size, 3), units="m/s")

        d.declare_partials("freestream_velocities", "alpha")
        d.declare_partials("freestream_velocities", "beta")
        d.declare_partials("freestream_velocities", "v")

        if rotational:
            nn = 3 * system_size
            row_col = np.arange(nn)
            val = np.ones((nn,))
            d.declare_partials("freestream_velocities", "rotational_velocities", rows=row_col, cols=row_col, val=val)
        return d.into(_spec)

    @cached_property
    def system_size(self) -> int:
        """Total number of VLM panels over all surfaces."""
        return vlm_system_size(self.surfaces)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Rotate the freestream velocities based on the angle of attack and the sideslip angle.
        alpha = inputs["alpha"][0] * np.pi / 180.0
        beta = inputs["beta"][0] * np.pi / 180.0

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        v_inf = inputs["v"][0] * np.array([cosa * cosb, -sinb, sina * cosb])
        outputs["freestream_velocities"][:, :] = v_inf

        if self.rotational:
            outputs["freestream_velocities"][:, :] += inputs["rotational_velocities"]

    def compute_partials(self, inputs, J):
        alpha = inputs["alpha"][0] * np.pi / 180.0
        beta = inputs["beta"][0] * np.pi / 180.0

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)

        J["freestream_velocities", "v"] = np.tile(np.array([cosa * cosb, -sinb, sina * cosb]), self.system_size)
        J["freestream_velocities", "alpha"] = np.tile(
            inputs["v"][0] * np.array([-sina * cosb, 0.0, cosa * cosb]) * np.pi / 180.0, self.system_size
        )
        J["freestream_velocities", "beta"] = np.tile(
            inputs["v"][0] * np.array([-cosa * sinb, -cosb, -sina * sinb]) * np.pi / 180.0, self.system_size
        )
