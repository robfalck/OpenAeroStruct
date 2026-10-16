from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class MonotonicConstraint(om.ExplicitComponent):
    """
    Produce a constraint that is violated if a user-chosen measure on the
    wing does not decrease monotonically from the root to the tip.

    Parameters
    ----------
    var_name : string
        The variable to which the user would like to apply the monotonic constraint.

    Returns
    -------
    monotonic[ny-1] : numpy array
        Values are greater than 0 if the constraint is violated.

    """

    var_name: str
    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        symmetry = f.surface["symmetry"]
        ny = f.surface["mesh"].shape[1]
        con_name = "monotonic_" + f.var_name
        d = VarDecl()

        d.add_input(f.var_name, val=np.zeros(ny))
        d.add_output(con_name, val=np.zeros(ny - 1))

        rows = np.arange(0, ny - 1)
        rows = np.vstack((rows, rows)).flatten(order="F")

        cols = np.arange(1, ny - 1)
        cols = np.vstack((cols, cols)).flatten(order="F")
        cols = np.insert(cols, 0, 0)
        cols = np.append(cols, ny - 1)

        sparse_val = np.ones_like(rows)
        sparse_val[1::2] = -1
        if not symmetry:
            if ny % 2 == 0:
                sparse_val[ny - 2 :] *= -1
            else:
                sparse_val[ny - 1 :] *= -1

        d.declare_partials(con_name, f.var_name, rows=rows, cols=cols, val=sparse_val)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Compute the difference between adjacent variable values
        x = inputs[self.var_name]
        diff = x[:-1] - x[1:]
        if self.surface["symmetry"]:
            outputs["monotonic_" + self.var_name] = diff
        else:
            ny2 = (x.size - 1) // 2
            outputs["monotonic_" + self.var_name] = np.concatenate([diff[:ny2], -diff[ny2:]])
