from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.utils import failure_criteria
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class FailureKS(om.ExplicitComponent):
    """
    Aggregate failure constraints from the structure.

    To simplify the optimization problem, we aggregate the individual
    elemental failure constraints using a Kreisselmeier-Steinhauser (KS)
    function.

    The KS function produces a smoother constraint than using a max() function
    to find the maximum point of failure, which produces a better-posed
    optimization problem.

    The rho parameter controls how conservatively the KS function aggregates
    the failure constraints. A lower value is more conservative while a greater
    value is more aggressive (closer approximation to the max() function).

    Parameters
    ----------
    vonmises : ny-1 x 2 numpy array
        von Mises stress magnitudes for each FEM element.
    tsaiwu_sr : ny-1 x 4 * num_plies numpy array
        Tsai-Wu strength ratios for each FEM element (ply at each critical element).

    Returns
    -------
    failure : float
        KS aggregation quantity obtained by combining the failure criteria
        for each FEM node. Used to simplify the optimization problem by
        reducing the number of constraints. This entity is defined for either
        failure criteria, vonmises or tsaiwu_sr.
    """

    surface: Surface
    rho: float = 100.0

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        use_composite, num_failure_criteria, _ = failure_criteria(f.surface)
        d = VarDecl()

        input_name = "tsaiwu_sr" if use_composite else "vonmises"
        units = None if use_composite else "N/m**2"
        d.add_input(input_name, val=np.zeros((ny - 1, num_failure_criteria)), units=units)
        d.add_output("failure", val=0.0)

        d.declare_partials("*", "*")
        return d.into(data)

    def _stress(self, inputs):
        use_composite, _, limit = failure_criteria(self.surface)
        return inputs["tsaiwu_sr" if use_composite else "vonmises"], limit

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        stress_array, limit = self._stress(inputs)

        fmax = np.max(stress_array / limit - 1)

        ks = 1 / self.rho * np.log(np.sum(np.exp(self.rho * (stress_array / limit - 1 - fmax))))
        outputs["failure"] = fmax + ks

    def compute_partials(self, inputs, partials):
        use_composite, _, _ = failure_criteria(self.surface)
        stress_array, limit = self._stress(inputs)

        fmax = np.max(stress_array / limit - 1)
        i, j = np.where((stress_array / limit - 1) == fmax)
        i, j = i[0], j[0]

        ksb = 1.0

        tempb0 = ksb / (self.rho * np.sum(np.exp(self.rho * (stress_array / limit - fmax - 1))))
        tempb = np.exp(self.rho * (stress_array / limit - fmax - 1)) * self.rho * tempb0
        fmaxb = ksb - np.sum(tempb)

        derivs = tempb / limit
        derivs[i, j] += fmaxb / limit

        partials["failure", "tsaiwu_sr" if use_composite else "vonmises"] = derivs.reshape(1, -1)
