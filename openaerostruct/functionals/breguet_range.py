from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class BreguetRange(om.ExplicitComponent):
    """
    Computes the fuel burn using the Breguet range equation using
    the computed CL, CD, weight, and provided specific fuel consumption, speed of sound,
    Mach number, initial weight, and range.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    CT : float
        Specific fuel consumption for the entire aircraft.
    speed_of_sound : float
        The Mach speed, speed of sound, at the specified flight condition.
    R : float
        The total range of the aircraft, used to backcalculate the fuel mass.
    Mach_number : float
        The Mach number of the aircraft at the specified flight condition.
    W0 : float
        The operating empty weight of the aircraft, without fuel or structural
        mass. Supplied in kg despite being a 'weight' due to convention.
    load_factor : float
        Multiplicative factor on gravity. 1.0 is normal flight; 2.5 would be
        for a 2.5g manuever.
    _structural_mass : float
        Weight of a single lifting surface's structural spar.

    Returns
    -------
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    """

    surfaces: list[Surface]

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data

        for surface in f["surfaces"]:
            name = surface["name"]
            d.add_input(name + "_structural_mass", val=1.0, units="kg")

        d.add_input("CT", val=0.25, units="1/s")
        d.add_input("CL", val=0.7)
        d.add_input("CD", val=0.02)
        d.add_input("speed_of_sound", val=100.0, units="m/s")
        d.add_input("R", val=3000.0, units="m")
        d.add_input("Mach_number", val=1.2)
        d.add_input("W0", val=200.0, units="kg")

        d.add_output("fuelburn", val=1.0, units="kg")

        d.declare_partials("*", "*")
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        CT = inputs["CT"]
        a = inputs["speed_of_sound"]
        R = inputs["R"]
        M = inputs["Mach_number"]
        W0 = inputs["W0"]

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = 0.0
        for surface in self.surfaces:
            name = surface["name"]
            Ws += inputs[name + "_structural_mass"]

        CL = inputs["CL"]
        CD = inputs["CD"]

        outputs["fuelburn"] = (W0 + Ws) * (np.exp(R * CT / a / M * CD / CL) - 1)

    def compute_partials(self, inputs, partials):
        CT = inputs["CT"]
        a = inputs["speed_of_sound"]
        R = inputs["R"]
        M = inputs["Mach_number"]
        W0 = inputs["W0"]

        Ws = 0.0
        for surface in self.surfaces:
            name = surface["name"]
            Ws += inputs[name + "_structural_mass"]

        CL = inputs["CL"]
        CD = inputs["CD"]

        dfb_dCL = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M * CD / CL**2
        dfb_dCD = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M / CL
        dfb_dCT = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R / a / M / CL * CD
        dfb_dR = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) / a / M / CL * CD * CT
        dfb_da = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a**2 / M * CD / CL
        dfb_dM = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M**2 * CD / CL

        dfb_dW = np.exp(R * CT / a / M * CD / CL) - 1

        partials["fuelburn", "CL"] = dfb_dCL
        partials["fuelburn", "CD"] = dfb_dCD
        partials["fuelburn", "CT"] = dfb_dCT
        partials["fuelburn", "speed_of_sound"] = dfb_da
        partials["fuelburn", "R"] = dfb_dR
        partials["fuelburn", "Mach_number"] = dfb_dM
        partials["fuelburn", "W0"] = dfb_dW

        for surface in self.surfaces:
            name = surface["name"]
            inp_name = name + "_structural_mass"
            partials["fuelburn", inp_name] = dfb_dW
