"""
Class definition for the MeshPointForces component.
"""

from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class MeshPointForces(om.ExplicitComponent):
    """
    Component that simply converts the forces on the panel to an equivalent set of forces at the
    mesh points.

    Here we just assume the leading edge points take slightly more of the load so that the centroid
    ends up at the quarter chord. The corresponding weights are stored in the le_wt and
    te_wt options.

    Parameters
    ----------
    sec_forces[nx-1, ny-1, 3] : numpy array
        The panel forces for each lifting surface.
        There is one of these per surface.

    Returns
    -------
    mesh_point_forces[nx, ny, 3] : numpy array
        The aeordynamic forces evaluated at the mesh nodes for each lifting surface.
        There is one of these per surface.
    """

    surfaces: list[Surface]
    le_wt: float = 0.75 * 0.5
    te_wt: float = 0.25 * 0.5

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surfaces = f["surfaces"]
        le_wt = f["le_wt"]
        te_wt = f["te_wt"]

        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            sec_forces_name = "{}_sec_forces".format(name)
            mesh_point_forces_name = "{}_mesh_point_forces".format(name)

            d.add_input(sec_forces_name, shape=(nx - 1, ny - 1, 3), units="N", tags=["mphys_coupling"])

            # TODO: what should res_ref be when it was np.sqrt(self.comm.size)
            d.add_output(mesh_point_forces_name, val=np.zeros((nx, ny, 3)), units="N", tags=["mphys_coupling"])

            # Sparse partials
            rowcol = np.arange(3 * (ny - 1))
            row2 = rowcol + 3

            rows1 = np.concatenate([rowcol, row2])
            cols1 = np.concatenate([rowcol, rowcol])

            le_rows = np.tile(rows1, nx - 1) + np.repeat(3 * ny * np.arange(nx - 1), 6 * (ny - 1))
            te_le_cols = np.tile(cols1, nx - 1) + np.repeat(3 * (ny - 1) * np.arange(nx - 1), 6 * (ny - 1))

            te_rows = le_rows + 3 * ny

            rows = np.concatenate([le_rows, te_rows])
            cols = np.concatenate([te_le_cols, te_le_cols])

            nn = len(rows)
            nn2 = int(nn / 2)
            vals = np.empty((nn,))

            vals[:nn2] = le_wt
            vals[nn2:] = te_wt

            d.declare_partials(mesh_point_forces_name, sec_forces_name, rows=rows, cols=cols, val=vals)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute the forces on the nodmesh points from the panel section force.
        """
        surfaces = self.surfaces

        le_wt = self.le_wt
        te_wt = self.te_wt
        for surface in surfaces:
            name = surface["name"]
            sec_forces_name = "{}_sec_forces".format(name)
            mesh_point_forces_name = "{}_mesh_point_forces".format(name)

            sec_forces = inputs[sec_forces_name]

            outputs[mesh_point_forces_name][:] = 0.0
            outputs[mesh_point_forces_name][:-1, :-1, :] += sec_forces * le_wt
            outputs[mesh_point_forces_name][1:, :-1, :] += sec_forces * te_wt
            outputs[mesh_point_forces_name][1:, 1:, :] += sec_forces * te_wt
            outputs[mesh_point_forces_name][:-1, 1:, :] += sec_forces * le_wt
