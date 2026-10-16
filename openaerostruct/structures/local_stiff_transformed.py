from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class LocalStiffTransformed(om.ExplicitComponent):
    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("transform", shape=(ny - 1, 12, 12))
        d.add_input("local_stiff_permuted", shape=(ny - 1, 12, 12))
        d.add_output("local_stiff_transformed", shape=(ny - 1, 12, 12))

        indices = np.arange(144 * (ny - 1)).reshape((ny - 1, 12, 12))
        ones = np.ones((12, 12), int)

        rows = np.einsum("ijk,lm->ijklm", indices, ones).flatten()
        cols = np.einsum("ilm,jk->ijklm", indices, ones).flatten()

        d.declare_partials("local_stiff_transformed", "transform", rows=rows, cols=cols)
        d.declare_partials("local_stiff_transformed", "local_stiff_permuted", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["local_stiff_transformed"] = np.einsum(
            "ilj,ilm,imk->ijk", inputs["transform"], inputs["local_stiff_permuted"], inputs["transform"]
        )

    def compute_partials(self, inputs, partials):
        partials["local_stiff_transformed", "local_stiff_permuted"] = np.einsum(
            "ilj,imk->ijklm", inputs["transform"], inputs["transform"]
        ).flatten()

        partials["local_stiff_transformed", "transform"] = (
            np.einsum("ilj,ilp,kq->ijkpq", inputs["transform"], inputs["local_stiff_permuted"], np.eye(12))
            + np.einsum("jq,ipm,imk->ijkpq", np.eye(12), inputs["local_stiff_permuted"], inputs["transform"])
        ).flatten()
