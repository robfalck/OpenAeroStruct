from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.utils import norm
from openaerostruct.utils.constants import grav_constant
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class StructureWeightLoads(om.ExplicitComponent):
    """
    Compute the nodal loads from the weight of the wing structure to be applied to the wing
    structure.

    Parameters
    ----------
    element_mass[ny-1] : numpy array
        Weight for each wing-structure segment.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    load_factor : float
        Load factor for the flight point.

    Returns
    -------
    struct_weight_loads[ny, 6] : numpy array
        Flattened array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    element_lengths[ny-1] : numpy array
        Lengths of the FEM finite elements.
    """

    surface: Surface
    check_partials_options: om.CheckPartialsOptions | None = om.CheckPartialsOptions(differentiator="cs")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("element_mass", val=np.zeros((ny - 1)), units="kg")
        d.add_input("nodes", val=np.zeros((ny, 3)), units="m")
        d.add_input("load_factor", val=1.0)

        d.add_output("struct_weight_loads", val=np.zeros((ny, 6)), units="N")
        d.add_output("element_lengths", val=np.zeros(ny - 1), units="m")

        rows, cols = cls._lf_pattern(ny)
        d.declare_partials("struct_weight_loads", "load_factor", rows=rows, cols=cols)

        rows, cols = cls._em_pattern(ny)
        d.declare_partials("struct_weight_loads", "element_mass", rows=rows, cols=cols)

        rows, cols = cls._nodes_pattern(ny)
        d.declare_partials("struct_weight_loads", "nodes", rows=rows, cols=cols)

        d.declare_partials(
            "element_lengths",
            "nodes",
            rows=np.repeat(np.arange(ny - 1), 6),
            cols=(3 * np.arange(ny - 1)[:, None] + np.arange(6)).flatten(),
        )
        return d.into(data)

    @staticmethod
    def _lf_pattern(ny):
        rows = (6 * np.arange(ny)[:, None] + np.array([2, 3, 4])).flatten()
        return rows, np.zeros_like(rows)

    @staticmethod
    def _em_pattern(ny):
        # Row (k, c) for c in z-force and the two bending moments depends on elements k-1 and k
        rows, cols = [], []
        for k in range(ny):
            for c in (2, 3, 4):
                for e in range(max(k - 1, 0), min(k + 1, ny - 1)):
                    rows.append(6 * k + c)
                    cols.append(e)
        return np.array(rows), np.array(cols)

    @staticmethod
    def _nodes_pattern(ny):
        # Bending-moment rows of node k depend on the coordinates of nodes k-1, k and k+1
        rows, cols = [], []
        for k in range(ny):
            for c in (3, 4):
                for col in range(3 * max(k - 1, 0), 3 * min(k + 2, ny)):
                    rows.append(6 * k + c)
                    cols.append(col)
        return np.array(rows), np.array(cols)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nodes = inputs["nodes"]
        ny = nodes.shape[0]
        lf = inputs["load_factor"].reshape(-1)[0]
        struct_weights = inputs["element_mass"] * lf * grav_constant

        deltas = nodes[1:, :] - nodes[:-1, :]
        element_lengths = norm(deltas, axis=1)
        del0 = deltas[:, 0]
        del1 = deltas[:, 1]

        # Assume weight coincides with the elastic axis
        z_forces_for_each = struct_weights / 2.0
        z_moments_for_each = struct_weights / 12.0 * (del0**2 + del1**2) ** 0.5

        loads = np.zeros((ny, 6), dtype=np.result_type(nodes, struct_weights))

        loads[:-1, 2] += -z_forces_for_each
        loads[1:, 2] += -z_forces_for_each

        # Bending moments for consistency
        bm3 = z_moments_for_each * del1 / element_lengths
        loads[:-1, 3] += -bm3
        loads[1:, 3] += bm3

        bm4 = z_moments_for_each * del0 / element_lengths
        loads[:-1, 4] += -bm4
        loads[1:, 4] += bm4

        outputs["struct_weight_loads"] = loads
        outputs["element_lengths"] = element_lengths

    def compute_partials(self, inputs, partials):
        nodes = inputs["nodes"]
        ny = nodes.shape[0]
        n = ny - 1
        lf = inputs["load_factor"].reshape(-1)[0]
        mass = inputs["element_mass"]
        weights = mass * lf * grav_constant

        deltas = nodes[1:, :] - nodes[:-1, :]
        d0, d1, d2 = deltas[:, 0], deltas[:, 1], deltas[:, 2]
        length = norm(deltas, axis=1)
        s = np.sqrt(d0**2 + d1**2)

        # Moments are w / 12 * d1 * r and w / 12 * d0 * r, with r = s / length
        r = s / length
        dr0 = d0 * (1.0 / (s * length) - s / length**3)
        dr1 = d1 * (1.0 / (s * length) - s / length**3)
        dr2 = -s * d2 / length**3

        # Derivatives of bm3 and bm4 per unit weight with respect to deltas
        dbm3 = np.stack([d1 * dr0, r + d1 * dr1, d1 * dr2], axis=1) / 12.0
        dbm4 = np.stack([r + d0 * dr0, d0 * dr1, d0 * dr2], axis=1) / 12.0
        bm3_unit = d1 * r / 12.0
        bm4_unit = d0 * r / 12.0

        # Derivative of the loads with respect to the element weights (per unit weight)
        # Rows are element start nodes (-) and end nodes (+)
        elem = np.arange(n)

        dloads_dw = np.zeros((ny * 6, n))
        dloads_dw[6 * elem + 2, elem] = -0.5
        dloads_dw[6 * (elem + 1) + 2, elem] = -0.5
        dloads_dw[6 * elem + 3, elem] = -bm3_unit
        dloads_dw[6 * (elem + 1) + 3, elem] = bm3_unit
        dloads_dw[6 * elem + 4, elem] = -bm4_unit
        dloads_dw[6 * (elem + 1) + 4, elem] = bm4_unit

        rows, cols = self._em_pattern(ny)
        partials["struct_weight_loads", "element_mass"] = dloads_dw[rows, cols] * lf * grav_constant

        rows, cols = self._lf_pattern(ny)
        partials["struct_weight_loads", "load_factor"] = (dloads_dw @ mass)[rows] * grav_constant

        dloads_dn = np.zeros((ny * 6, ny * 3))
        for e in range(n):
            lo, hi = 3 * e, 3 * (e + 1)
            for comp, db in ((3, dbm3[e]), (4, dbm4[e])):
                wdb = weights[e] * db
                # loads of the start node are -bm and of the end node +bm; bm depends on x[e+1] - x[e]
                dloads_dn[6 * e + comp, lo : lo + 3] += wdb
                dloads_dn[6 * e + comp, hi : hi + 3] -= wdb
                dloads_dn[6 * (e + 1) + comp, lo : lo + 3] -= wdb
                dloads_dn[6 * (e + 1) + comp, hi : hi + 3] += wdb
        rows, cols = self._nodes_pattern(ny)
        partials["struct_weight_loads", "nodes"] = dloads_dn[rows, cols]

        unit = deltas / length[:, None]
        partials["element_lengths", "nodes"] = np.hstack((-unit, unit)).flatten()
