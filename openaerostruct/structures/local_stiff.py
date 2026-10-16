from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


coeffs_2 = np.array(
    [
        [1.0, -1.0],
        [-1.0, 1.0],
    ]
)

coeffs_y = np.array(
    [
        [12.0, -6.0, -12.0, -6.0],
        [-6.0, 4.0, 6.0, 2.0],
        [-12.0, 6.0, 12.0, 6.0],
        [-6.0, 2.0, 6.0, 4.0],
    ]
)

coeffs_z = np.array(
    [
        [12.0, 6.0, -12.0, 6.0],
        [6.0, 4.0, -6.0, 2.0],
        [-12.0, -6.0, 12.0, -6.0],
        [6.0, 2.0, -6.0, 4.0],
    ]
)


class LocalStiff(om.ExplicitComponent):
    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("A", shape=ny - 1, units="m**2")
        d.add_input("J", shape=ny - 1, units="m**4")
        d.add_input("Iy", shape=ny - 1, units="m**4")
        d.add_input("Iz", shape=ny - 1, units="m**4")
        d.add_input("element_lengths", shape=ny - 1, units="m")

        d.add_output("local_stiff", shape=(ny - 1, 12, 12))

        rows = np.arange(144 * (ny - 1))
        cols = np.outer(np.arange(ny - 1), np.ones(144, int)).flatten()

        d.declare_partials("local_stiff", "A", rows=rows, cols=cols)
        d.declare_partials("local_stiff", "J", rows=rows, cols=cols)
        d.declare_partials("local_stiff", "Iy", rows=rows, cols=cols)
        d.declare_partials("local_stiff", "Iz", rows=rows, cols=cols)
        d.declare_partials("local_stiff", "element_lengths", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        E = self.surface["E"]
        G = self.surface["G"]

        A = inputs["A"]
        Iy = inputs["Iy"]
        Iz = inputs["Iz"]
        J = inputs["J"]
        L = inputs["element_lengths"]

        out = np.zeros((len(L), 12, 12), dtype=np.result_type(A, Iy, Iz, J, L))

        for i in range(2):
            for j in range(2):
                out[:, 0 + i, 0 + j] = E * A / L * coeffs_2[i, j]
                out[:, 2 + i, 2 + j] = G * J / L * coeffs_2[i, j]

        for i in range(4):
            for j in range(4):
                out[:, 4 + i, 4 + j] = E * Iy / L**3 * coeffs_y[i, j]
                out[:, 8 + i, 8 + j] = E * Iz / L**3 * coeffs_z[i, j]

        for i in [1, 3]:
            for j in range(4):
                out[:, 4 + i, 4 + j] *= L
                out[:, 8 + i, 8 + j] *= L
        for i in range(4):
            for j in [1, 3]:
                out[:, 4 + i, 4 + j] *= L
                out[:, 8 + i, 8 + j] *= L
        outputs["local_stiff"] = out

    def compute_partials(self, inputs, partials):
        E = self.surface["E"]
        G = self.surface["G"]

        A = inputs["A"]
        Iy = inputs["Iy"]
        Iz = inputs["Iz"]
        J = inputs["J"]
        L = inputs["element_lengths"]

        n = len(L)
        derivs_A = np.zeros((n, 12, 12))
        derivs_Iy = np.zeros((n, 12, 12))
        derivs_Iz = np.zeros((n, 12, 12))
        derivs_J = np.zeros((n, 12, 12))
        derivs_L = np.zeros((n, 12, 12))

        for i in range(2):
            for j in range(2):
                derivs_A[:, 0 + i, 0 + j] = E / L * coeffs_2[i, j]
                derivs_L[:, 0 + i, 0 + j] = -E * A / L**2 * coeffs_2[i, j]

                derivs_J[:, 2 + i, 2 + j] = G / L * coeffs_2[i, j]
                derivs_L[:, 2 + i, 2 + j] = -G * J / L**2 * coeffs_2[i, j]

        for i in range(4):
            for j in range(4):
                derivs_Iy[:, 4 + i, 4 + j] = E / L**3 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -3 * E * Iy / L**4 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = E / L**3 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -3 * E * Iz / L**4 * coeffs_z[i, j]

        for i in [1, 3]:
            for j in range(4):
                derivs_Iy[:, 4 + i, 4 + j] = E / L**2 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -2 * E * Iy / L**3 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = E / L**2 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -2 * E * Iz / L**3 * coeffs_z[i, j]
        for i in range(4):
            for j in [1, 3]:
                derivs_Iy[:, 4 + i, 4 + j] = E / L**2 * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -2 * E * Iy / L**3 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = E / L**2 * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -2 * E * Iz / L**3 * coeffs_z[i, j]

        for i in [1, 3]:
            for j in [1, 3]:
                derivs_Iy[:, 4 + i, 4 + j] = E / L * coeffs_y[i, j]
                derivs_L[:, 4 + i, 4 + j] = -E * Iy / L**2 * coeffs_y[i, j]

                derivs_Iz[:, 8 + i, 8 + j] = E / L * coeffs_z[i, j]
                derivs_L[:, 8 + i, 8 + j] = -E * Iz / L**2 * coeffs_z[i, j]

        partials["local_stiff", "A"] = derivs_A.flatten()
        partials["local_stiff", "Iy"] = derivs_Iy.flatten()
        partials["local_stiff", "Iz"] = derivs_Iz.flatten()
        partials["local_stiff", "J"] = derivs_J.flatten()
        partials["local_stiff", "element_lengths"] = derivs_L.flatten()
