from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_norm, compute_norm_deriv
from openaerostruct.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2


class Transform(om.ExplicitComponent):
    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("nodes", shape=(ny, 3), units="m")
        d.add_output("transform", shape=(ny - 1, 12, 12))

        mesh_indices = np.arange(3 * ny).reshape((ny, 3))
        transform_indices = np.arange(144 * (ny - 1)).reshape((ny - 1, 12, 12))

        rows = np.concatenate(
            [
                np.einsum("ijk,l->ijkl", transform_indices, np.ones(3, int)).flatten(),
                np.einsum("ijk,l->ijkl", transform_indices, np.ones(3, int)).flatten(),
            ]
        )
        cols = np.concatenate(
            [
                np.einsum("il,jk->ijkl", mesh_indices[:-1, :], np.ones((12, 12), int)).flatten(),
                np.einsum("il,jk->ijkl", mesh_indices[1:, :], np.ones((12, 12), int)).flatten(),
            ]
        )
        d.declare_partials("transform", "nodes", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ny = self.surface["mesh"].shape[1]
        ref_axis = np.outer(np.ones(ny - 1), np.array([1.0, 0.0, 0.0]))
        P0 = inputs["nodes"][:-1, :]
        P1 = inputs["nodes"][1:, :]
        norm = compute_norm(P1 - P0)
        row0 = (P1 - P0) / norm

        cross = compute_cross(row0, ref_axis)
        norm = compute_norm(cross)
        row1 = cross / norm

        cross = compute_cross(row0, row1)
        row2 = cross

        out = np.zeros((ny - 1, 12, 12), dtype=np.result_type(row0, row1, row2))
        for k in range(4):
            out[:, 3 * k + 0, 3 * k : 3 * k + 3] = row0
            out[:, 3 * k + 1, 3 * k : 3 * k + 3] = row1
            out[:, 3 * k + 2, 3 * k : 3 * k + 3] = row2
        outputs["transform"] = out

    def compute_partials(self, inputs, partials):
        ny = self.surface["mesh"].shape[1]
        ref_axis = np.outer(np.ones(ny - 1), np.array([1.0, 0.0, 0.0]))

        P0 = inputs["nodes"][:-1, :]
        P1 = inputs["nodes"][1:, :]
        P_deriv = np.einsum("i,jk->ijk", np.ones(ny - 1), np.eye(3))
        norm = compute_norm(P1 - P0)
        norm_deriv = compute_norm_deriv(P1 - P0, P_deriv)
        row0 = (P1 - P0) / norm
        row0_deriv = P_deriv / add_ones_axis(norm) - add_ones_axis(P1 - P0) / add_ones_axis(norm) ** 2 * norm_deriv

        cross = compute_cross(row0, ref_axis)
        cross_deriv = compute_cross_deriv1(row0_deriv, ref_axis)
        norm = compute_norm(cross)
        norm_deriv = compute_norm_deriv(cross, cross_deriv)
        row1 = cross / norm
        row1_deriv = cross_deriv / add_ones_axis(norm) - add_ones_axis(cross) / add_ones_axis(norm) ** 2 * norm_deriv

        cross = compute_cross(row0, row1)
        cross_deriv = compute_cross_deriv1(row0_deriv, row1) + compute_cross_deriv2(row0, row1_deriv)
        row2_deriv = cross_deriv

        derivs = np.zeros((2, ny - 1, 12, 12, 3))

        for k in range(4):
            derivs[0, :, 3 * k + 0, 3 * k : 3 * k + 3, :] = -row0_deriv
            derivs[1, :, 3 * k + 0, 3 * k : 3 * k + 3, :] = row0_deriv

            derivs[0, :, 3 * k + 1, 3 * k : 3 * k + 3, :] = -row1_deriv
            derivs[1, :, 3 * k + 1, 3 * k : 3 * k + 3, :] = row1_deriv

            derivs[0, :, 3 * k + 2, 3 * k : 3 * k + 3, :] = -row2_deriv
            derivs[1, :, 3 * k + 2, 3 * k : 3 * k + 3, :] = row2_deriv

        partials["transform", "nodes"] = derivs.flatten()
