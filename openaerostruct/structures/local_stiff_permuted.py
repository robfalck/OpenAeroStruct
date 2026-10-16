from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


row_indices = np.arange(12)
col_indices = np.array(
    [
        0,
        6,
        3,
        9,
        2,
        4,
        8,
        10,
        1,
        5,
        7,
        11,
    ]
)
mtx = np.zeros((12, 12))
mtx[row_indices, col_indices] = 1.0


class LocalStiffPermuted(om.ExplicitComponent):
    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("local_stiff", shape=(ny - 1, 12, 12))
        d.add_output("local_stiff_permuted", shape=(ny - 1, 12, 12))

        indices = np.arange(144 * (ny - 1)).reshape((ny - 1, 12, 12))
        ones = np.ones((12, 12), int)

        vals = np.einsum("i,jl,mk->ijklm", np.ones(ny - 1), mtx.T, mtx)

        vals = vals.flatten()
        rows = np.einsum("ijk,lm->ijklm", indices, ones).flatten()
        cols = np.einsum("ilm,jk->ijklm", indices, ones).flatten()
        d.declare_partials("local_stiff_permuted", "local_stiff", val=vals, rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["local_stiff_permuted"] = np.einsum("jl,ilm,mk->ijk", mtx.T, inputs["local_stiff"], mtx)
