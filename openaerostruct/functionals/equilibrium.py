from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.constants import grav_constant
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class Equilibrium(om.ExplicitComponent):
    """
    Computes L_equals_W, which is a normalized measure of the weight of the
    aircraft minus the total generated lift. So if L_equals_W is positive,
    the aircraft weighs more than the amount of lift it's producing.

    Generally we use this measure to ensure that lift equals weight for a cruise
    fligth condition.
    Note that we add information from each lifting surface.

    Parameters
    ----------
    structural_mass : float
        Total weight of the structural spar for a given surface.

    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.
    W0 : float
        The operating empty weight of the aircraft, without fuel or structural
        mass. Supplied in kg despite being a 'weight' due to convention.
    load_factor : float
        Multiplicative factor on gravity. 1.0 is normal flight; 2.5 would be
        for a 2.5g manuever.
    CL : float
        Total coefficient of lift (CL) for the entire aircraft.
    S_ref_total : float
        Total surface area of the aircraft based on the sum of individual
        surface areas.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    L_equals_W : float
        Equality constraint for lift = total weight. L_equals_W = 0 for the
        constraint to be satisfied.
    total_weight : float
        Total weight of the entire aircraft, including W0, all structural
        weights, and fuel.
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
            d.declare_partials("L_equals_W", name + "_structural_mass")
            d.declare_partials("total_weight", name + "_structural_mass")

        d.add_input("fuelburn", val=123.0, units="kg")
        d.add_input("W0", val=1000.0, units="kg")
        d.add_input("load_factor", val=1.05)

        d.add_input("CL", val=0.7)

        d.add_input("S_ref_total", val=15.0, units="m**2")
        d.add_input("v", val=100.0, units="m/s")
        d.add_input("rho", val=1.2, units="kg/m**3")

        d.add_output("L_equals_W", val=1.0)
        d.add_output("total_weight", val=1.0, units="N")

        d.declare_partials("L_equals_W", "CL")
        d.declare_partials("L_equals_W", "S_ref_total")
        d.declare_partials("L_equals_W", "W0")
        d.declare_partials("L_equals_W", "fuelburn")
        d.declare_partials("L_equals_W", "load_factor")
        d.declare_partials("L_equals_W", "rho")
        d.declare_partials("L_equals_W", "v")
        d.declare_partials("total_weight", "W0")
        d.declare_partials("total_weight", "fuelburn")
        d.declare_partials("total_weight", "load_factor")
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = grav_constant * inputs["load_factor"]
        W0 = inputs["W0"]
        rho = inputs["rho"]
        v = inputs["v"]

        structural_mass = 0.0
        for surface in self.surfaces:
            name = surface["name"]
            structural_mass += inputs[name + "_structural_mass"]

        S_ref_tot = inputs["S_ref_total"]

        tot_weight = (structural_mass + inputs["fuelburn"] + W0) * g

        outputs["total_weight"] = tot_weight
        outputs["L_equals_W"] = 1 - (0.5 * rho * v**2 * S_ref_tot) * inputs["CL"] / tot_weight

    def compute_partials(self, inputs, partials):
        g = grav_constant * inputs["load_factor"]
        W0 = inputs["W0"]
        rho = inputs["rho"]
        v = inputs["v"]

        structural_mass = 0.0
        for surface in self.surfaces:
            name = surface["name"]
            structural_mass += inputs[name + "_structural_mass"]

        S_ref_tot = inputs["S_ref_total"]

        tot_weight = (structural_mass + inputs["fuelburn"] + W0) * g

        L = inputs["CL"] * (0.5 * rho * v**2 * S_ref_tot)

        partials["total_weight", "fuelburn"] = g
        partials["total_weight", "W0"] = g
        partials["total_weight", "load_factor"] = (inputs["fuelburn"] + inputs["W0"] + structural_mass) * grav_constant

        partials["L_equals_W", "fuelburn"] = L / tot_weight**2 * g
        partials["L_equals_W", "W0"] = L / tot_weight**2 * g
        partials["L_equals_W", "load_factor"] = (
            L / tot_weight**2 * (inputs["fuelburn"] + inputs["W0"] + structural_mass) * grav_constant
        )
        partials["L_equals_W", "rho"] = -0.5 * S_ref_tot * v**2 * inputs["CL"] / tot_weight
        partials["L_equals_W", "v"] = -rho * S_ref_tot * v * inputs["CL"] / tot_weight
        partials["L_equals_W", "CL"] = -0.5 * rho * v**2 * S_ref_tot / tot_weight
        partials["L_equals_W", "S_ref_total"] = -0.5 * rho * v**2 * inputs["CL"] / tot_weight

        for surface in self.surfaces:
            name = surface["name"]
            partials["total_weight", name + "_structural_mass"] = 1.0 * g
            partials["L_equals_W", name + "_structural_mass"] = L / tot_weight**2 * g
