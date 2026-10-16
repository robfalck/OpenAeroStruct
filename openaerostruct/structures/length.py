from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.utils import norm
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface
from openaerostruct.utils.vector_algebra import compute_norm


class Length(om.ExplicitComponent):
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
        d.add_output("element_lengths", shape=ny - 1, units="m")

        mesh_indices = np.arange(3 * ny).reshape((ny, 3))
        length_indices = np.arange(ny - 1)

        rows = np.tile(np.outer(length_indices, np.ones(3, int)).flatten(), 2)
        cols = np.concatenate(
            [
                mesh_indices[:-1, :].flatten(),
                mesh_indices[1:, :].flatten(),
            ]
        )
        d.declare_partials("element_lengths", "nodes", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vec = inputs["nodes"][1:, :] - inputs["nodes"][:-1, :]

        outputs["element_lengths"] = norm(vec, axis=-1)

    def compute_partials(self, inputs, partials):
        vec = inputs["nodes"][1:, :] - inputs["nodes"][:-1, :]

        derivs = np.empty((2, vec.shape[0], 3))
        derivs[0, :, :] = -vec / compute_norm(vec)
        derivs[1, :, :] = vec / compute_norm(vec)
        partials["element_lengths", "nodes"] = derivs.flatten()
