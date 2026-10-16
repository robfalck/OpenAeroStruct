from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives.complex_step import ComplexStep
from om4.derivatives.differentiator import Differentiator
from om4.utils.polymorphic import Polymorphic

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class WingboxFuelVolDelta(om.ExplicitComponent):
    """
    Create a constraint to ensure the wingbox has enough internal volume to
    store the required fuel.

    Parameters
    ----------
    fuelburn : float
        Fuel weight
    fuel_vols[ny-1] : numpy array
        The magnitude of each individual panel's fuel-carrying volumes.

    Returns
    -------
    fuel_vol_delta : float
        If the value is negative, then there isn't enough volume for the fuel.
    """

    surface: Surface
    differentiator: Polymorphic[Differentiator] | None = ComplexStep()

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("fuelburn", val=0.0, units="kg")
        d.add_input("fuel_vols", val=np.zeros((ny - 1)), units="m**3")
        d.add_output("fuel_vol_delta", val=0.0, units="m**3")

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_weight = inputs["fuelburn"]
        reserves = self.surface["Wf_reserve"]
        fuel_density = self.surface["fuel_density"]

        if self.surface["symmetry"] is True:
            fuel_weight = fuel_weight / 2.0
            reserves = reserves / 2.0

        sum_vols = np.sum(inputs["fuel_vols"])

        # This is used for the fuel-volume constraint. It should be positive for fuel to fit.
        outputs["fuel_vol_delta"] = sum_vols - (fuel_weight + reserves) / fuel_density
