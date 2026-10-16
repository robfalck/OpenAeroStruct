from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.compute_nodes import get_fem_origin
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


def _sec_forces_pattern(nx, ny):
    # Direct loads wrt sec_forces.
    base_row = np.array([0, 1, 2, 6, 7, 8])
    base_col = np.array([0, 1, 2, 0, 1, 2])
    row = np.tile(base_row, ny - 1) + np.repeat(6 * np.arange(ny - 1), 6)
    col = np.tile(base_col, ny - 1) + np.repeat(3 * np.arange(ny - 1), 6)
    rows1 = np.tile(row, nx - 1)
    cols1 = np.tile(col, nx - 1) + np.repeat(3 * (ny - 1) * np.arange(nx - 1), 6 * (ny - 1))

    # Cross product terms.
    base_row = np.array([3, 3, 4, 4, 5, 5])
    base_col = np.array([1, 2, 0, 2, 0, 1])
    row = np.tile(base_row, ny - 1) + np.repeat(6 * np.arange(ny - 1), 6)
    col = np.tile(base_col, ny - 1) + np.repeat(3 * np.arange(ny - 1), 6)
    row1 = np.tile(row, nx - 1)
    col1 = np.tile(col, nx - 1) + np.repeat(3 * (ny - 1) * np.arange(nx - 1), 6 * (ny - 1))
    rows2 = np.tile(row1, 2) + np.repeat(np.array([0, 6]), 6 * (nx - 1) * (ny - 1))
    cols2 = np.tile(col1, 2)

    return np.concatenate([rows1, rows2]), np.concatenate([cols1, cols2])


def _def_mesh_pattern(nx, ny):
    # Top diagonal is forward-most mesh point.
    base_row = np.array([3, 3, 4, 4, 5, 5])
    base_col = np.array([4, 5, 3, 5, 3, 4])
    row = np.tile(base_row, ny - 1) + np.repeat(6 * np.arange(ny - 1), 6)
    col = np.tile(base_col, ny - 1) + np.repeat(3 * np.arange(ny - 1), 6)
    rows1 = np.tile(row, nx)
    cols1 = np.tile(col, nx) + np.repeat(3 * ny * np.arange(nx), 6 * (ny - 1))

    # Bottom diagonal is backward-most mesh point.
    base_row = np.array([9, 9, 10, 10, 11, 11])
    base_col = np.array([1, 2, 0, 2, 0, 1])
    row = np.tile(base_row, ny - 1) + np.repeat(6 * np.arange(ny - 1), 6)
    col = np.tile(base_col, ny - 1) + np.repeat(3 * np.arange(ny - 1), 6)
    rows2 = np.tile(row, nx)
    cols2 = np.tile(col, nx) + np.repeat(3 * ny * np.arange(nx), 6 * (ny - 1))

    # Central diagonal blocks
    base_row = np.array([3, 3, 4, 4, 5, 5])
    base_col = np.array([1, 2, 0, 2, 0, 1])
    row = np.tile(base_row, ny) + np.repeat(6 * np.arange(ny), 6)
    col = np.tile(base_col, ny) + np.repeat(3 * np.arange(ny), 6)
    rows3 = np.tile(row, nx)
    cols3 = np.tile(col, nx) + np.repeat(3 * ny * np.arange(nx), 6 * ny)

    return np.concatenate([rows1, rows2, rows3]), np.concatenate([cols1, cols2, cols3])


class LoadTransfer(om.ExplicitComponent):
    """
    Perform aerodynamic load transfer.

    Apply the computed sectional forces on the aerodynamic surfaces to
    obtain the deformed mesh FEM loads.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the lifting surfaces after deformation.
        Arrays will be flattened in Fortran order (only relevant when more than one chordwise panel).
    sec_forces[nx-1, ny-1, 3] : numpy array
        Array containing the sectional forces acting on each panel.

    Returns
    -------
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component at each node,
        computed from the sectional forces. The first 3 columns are N, and the last 3 are N*m.
    """

    surface: Surface
    check_partials_options: om.CheckPartialsOptions | None = om.CheckPartialsOptions(differentiator="cs")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        nx, ny = f.surface["mesh"].shape[:2]
        d = VarDecl()

        d.add_input("def_mesh", val=np.zeros((nx, ny, 3)), units="m")
        d.add_input("sec_forces", val=np.zeros((nx - 1, ny - 1, 3)), units="N")

        # The first 3 columns are N and the last 3 are N*m.
        d.add_output("loads", val=np.zeros((ny, 6)), units="N")

        rows, cols = _sec_forces_pattern(nx, ny)
        d.declare_partials("loads", "sec_forces", rows=rows, cols=cols)
        rows, cols = _def_mesh_pattern(nx, ny)
        d.declare_partials("loads", "def_mesh", rows=rows, cols=cols)
        return d.into(data)

    def _weights(self):
        w1 = 0.25
        w2 = float(get_fem_origin(self.surface))
        return w1, w2

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mesh = inputs["def_mesh"]  # [nx, ny, 3]
        sec_forces = inputs["sec_forces"]
        w1, w2 = self._weights()
        dtype = np.result_type(mesh.dtype, sec_forces.dtype)
        loads = np.zeros((mesh.shape[1], 6), dtype=dtype)

        # The aero force acting on each panel is evenly transferred to the adjacent FEM nodes.
        sec_forces_sum = 0.5 * np.sum(sec_forces, axis=0)
        loads[:-1, :3] = sec_forces_sum
        loads[1:, :3] += sec_forces_sum

        # Aerodynamic centers at the quarter-chord point of each panel [nx-1, ny-1, 3]
        a_pts = (
            0.5 * (1 - w1) * mesh[:-1, :-1, :]
            + 0.5 * w1 * mesh[1:, :-1, :]
            + 0.5 * (1 - w1) * mesh[:-1, 1:, :]
            + 0.5 * w1 * mesh[1:, 1:, :]
        )

        # Structural nodes based on the fem_origin location [ny, 3]
        s_pts = (1 - w2) * mesh[0, :, :] + w2 * mesh[-1, :, :]

        # Moment contribution of sec_forces (acting on aero center) to the inner/outer adjacent node
        moment_in = np.sum(np.cross(a_pts - s_pts[:-1, :], 0.5 * sec_forces), axis=0)  # [ny-1, 3]
        moment_out = np.sum(np.cross(a_pts - s_pts[1:, :], 0.5 * sec_forces), axis=0)

        loads[:-1, 3:] = moment_in
        loads[1:, 3:] += moment_out
        outputs["loads"] = loads

    def compute_partials(self, inputs, partials):
        mesh = inputs["def_mesh"]
        sec_forces = inputs["sec_forces"]
        nx, ny = mesh.shape[:2]
        w1, w2 = self._weights()

        # Compute the aerodynamic centers at the quarter-chord point of each panel
        a_pts = (
            0.5 * (1 - w1) * mesh[:-1, :-1, :]
            + 0.5 * w1 * mesh[1:, :-1, :]
            + 0.5 * (1 - w1) * mesh[:-1, 1:, :]
            + 0.5 * w1 * mesh[1:, 1:, :]
        )

        # Compute the structural nodes
        s_pts = (1 - w2) * mesh[0, :, :] + w2 * mesh[-1, :, :]

        # ----- 1. dmoment__dsec_forces -----
        # Sensitivity of loads (moments) at inner node wrt sec_force
        diff_in = 0.5 * (a_pts - s_pts[:-1, :])  # moment arm from inner node to aero center.

        dmom_dsec_in = np.empty((nx - 1, ny - 1, 6))
        dmom_dsec_in[:, :, 0] = -diff_in[:, :, 2]
        dmom_dsec_in[:, :, 1] = diff_in[:, :, 1]
        dmom_dsec_in[:, :, 2] = diff_in[:, :, 2]
        dmom_dsec_in[:, :, 3] = -diff_in[:, :, 0]
        dmom_dsec_in[:, :, 4] = -diff_in[:, :, 1]
        dmom_dsec_in[:, :, 5] = diff_in[:, :, 0]

        # Repeat for moments at outer node wrt sec_force
        diff_out = 0.5 * (a_pts - s_pts[1:, :])  # moment arm from outer node to aero center.
        dmom_dsec_out = np.empty((nx - 1, ny - 1, 6))
        dmom_dsec_out[:, :, 0] = -diff_out[:, :, 2]
        dmom_dsec_out[:, :, 1] = diff_out[:, :, 1]
        dmom_dsec_out[:, :, 2] = diff_out[:, :, 2]
        dmom_dsec_out[:, :, 3] = -diff_out[:, :, 0]
        dmom_dsec_out[:, :, 4] = -diff_out[:, :, 1]
        dmom_dsec_out[:, :, 5] = diff_out[:, :, 0]

        id1 = 6 * (ny - 1) * (nx - 1)
        d_sec = np.empty(3 * id1)
        d_sec[:id1] = 0.5

        id2 = id1 * 2
        dmom_dsec_in = dmom_dsec_in.flatten()
        dmom_dsec_out = dmom_dsec_out.flatten()
        d_sec[id1:id2] = dmom_dsec_in
        d_sec[id2:] = dmom_dsec_out

        # ----- 2. dmoment__dmesh -----
        # Sensitivity of moments at inner nodes wrt diff_in (upper diagonal)
        dmom_ddiff_in = np.zeros((nx - 1, ny - 1, 6))
        dmom_ddiff_in[:, :, 0] = sec_forces[:, :, 2]
        dmom_ddiff_in[:, :, 1] = -sec_forces[:, :, 1]
        dmom_ddiff_in[:, :, 2] = -sec_forces[:, :, 2]
        dmom_ddiff_in[:, :, 3] = sec_forces[:, :, 0]
        dmom_ddiff_in[:, :, 4] = sec_forces[:, :, 1]
        dmom_ddiff_in[:, :, 5] = -sec_forces[:, :, 0]
        dmom_ddiff_in_sum = np.sum(dmom_ddiff_in, axis=0)

        # Sensitivity of moments at outer nodes wrt diff_out (lower diagonal)
        dmom_ddiff_out = np.zeros((nx - 1, ny - 1, 6))
        dmom_ddiff_out[:, :, 0] = sec_forces[:, :, 2]
        dmom_ddiff_out[:, :, 1] = -sec_forces[:, :, 1]
        dmom_ddiff_out[:, :, 2] = -sec_forces[:, :, 2]
        dmom_ddiff_out[:, :, 3] = sec_forces[:, :, 0]
        dmom_ddiff_out[:, :, 4] = sec_forces[:, :, 1]
        dmom_ddiff_out[:, :, 5] = -sec_forces[:, :, 0]
        dmom_ddiff_out_sum = np.sum(dmom_ddiff_out, axis=0)

        dmon_ddiff_diag = np.zeros((nx - 1, ny, 6))
        dmon_ddiff_diag[:, 1:, :] = dmom_ddiff_out
        dmon_ddiff_diag[:, :-1, :] += dmom_ddiff_in
        dmon_ddiff_diag_sum = np.zeros((1, ny, 6))
        dmon_ddiff_diag_sum[:, :-1, :] = dmom_ddiff_in_sum
        dmon_ddiff_diag_sum[:, 1:, :] += dmom_ddiff_out_sum

        dmom_ddiff_in = dmom_ddiff_in.flatten()
        dmom_ddiff_out = dmom_ddiff_out.flatten()
        dmon_ddiff_diag = dmon_ddiff_diag.flatten()
        dmon_ddiff_diag_sum = dmon_ddiff_diag_sum.flatten()

        idy = 6 * (ny - 1)
        idx = idy * nx
        idw = idy * (nx - 1)

        # Assignments overlap, so start from zero.
        d_mesh = np.zeros(2 * idx + 6 * ny * nx)

        # Upper diagonal blocks
        d_mesh[:idw] = dmom_ddiff_in * ((1 - w1) * 0.25)
        d_mesh[idy:idx] += dmom_ddiff_in * (w1 * 0.25)

        # Lower Diagonal blocks
        id2 = idx * 2
        d_mesh[idx : idx + idw] = dmom_ddiff_out * ((1 - w1) * 0.25)
        d_mesh[idx + idy : id2] += dmom_ddiff_out * (w1 * 0.25)

        # Central Diagonal blocks
        idy = 6 * ny
        idz = 6 * (nx - 1)
        id3 = id2 + idw + idz
        d_mesh[id2:id3] = dmon_ddiff_diag * ((1 - w1) * 0.25)
        d_mesh[id2 : id2 + idy] -= dmon_ddiff_diag_sum * ((1 - w2) * 0.5)

        id2 += idy
        id3 += idy
        d_mesh[id2:id3] += dmon_ddiff_diag * (w1 * 0.25)
        d_mesh[id3 - idy : id3] -= dmon_ddiff_diag_sum * (w2 * 0.5)

        partials["loads", "sec_forces"] = d_sec
        partials["loads", "def_mesh"] = d_mesh
