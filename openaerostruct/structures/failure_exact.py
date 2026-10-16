from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.utils import failure_criteria
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class FailureExact(om.ExplicitComponent):
    """
    Output individual failure constraints on each FEM element.

    Parameters
    ----------
    vonmises : ny-1 x 2 numpy array
        von Mises stress magnitudes for each FEM element.
    tsaiwu_sr : ny-1 x 4 * num_plies numpy array
        Tsai-Wu strength ratios for each FEM element (ply at each critical element).

    Returns
    -------
    failure : ny-1 x num_failure_criteria array
        Array of failure conditions. Positive if element has failed. This entity is defined for either failure criteria,
        vonmises or tsaiwu_sr. num_failure_criteria is 2 for tube, 4 for the isotropic wingbox and 4*num_plies for the
        composite wingbox.

    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        use_composite, num_failure_criteria, limit = failure_criteria(f.surface)
        d = VarDecl()

        if use_composite:
            d.add_input("tsaiwu_sr", val=np.zeros((ny - 1, num_failure_criteria)), units=None)
        else:
            d.add_input("vonmises", val=np.zeros((ny - 1, num_failure_criteria)), units="N/m**2")

        d.add_output("failure", val=np.zeros((ny - 1, num_failure_criteria)))

        n = (ny - 1) * num_failure_criteria
        d.declare_partials(
            "failure", "tsaiwu_sr" if use_composite else "vonmises", rows=np.arange(n), cols=np.arange(n), val=1 / limit
        )
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        use_composite, _, limit = failure_criteria(self.surface)
        stress = inputs["tsaiwu_sr"] if use_composite else inputs["vonmises"]
        outputs["failure"] = stress / limit - 1
