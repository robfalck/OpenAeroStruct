from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class TotalLoads(om.ExplicitComponent):
    """
    Add the loads from the aerodynamics, structural weight, and fuel weight.

    Parameters
    ----------
    loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the sectional forces.
    struct_weight_loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the weight of the wing-structure segments.
    fuel_weight_loads[ny, 6] : numpy array
        Array containing the loads applied on the FEM component,
        computed from the weight of the fuel.
    loads_from_point_masses[ny, 6] : numpy array
        The cumulative loads from all point masses.

    Returns
    -------
    total_loads[ny, 6] : numpy array
        Flattened array containing the total loads applied on the FEM.
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        surface = f.surface
        ny = surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("loads", val=np.ones((ny, 6)), units="N")
        if surface["struct_weight_relief"]:
            d.add_input("struct_weight_loads", val=np.zeros((ny, 6)), units="N")
        if surface["distributed_fuel_weight"]:
            d.add_input("fuel_weight_loads", val=np.zeros((ny, 6)), units="N")
        if "n_point_masses" in surface.keys():
            d.add_input("loads_from_point_masses", val=np.zeros((ny, 6)), units="N")
            d.add_input("loads_from_thrusts", val=np.zeros((ny, 6)), units="N")

        d.add_output("total_loads", val=np.ones((ny, 6)), units="N")

        arange = np.arange(ny * 6)

        d.declare_partials("total_loads", "loads", rows=arange, cols=arange, val=1.0)

        if surface["struct_weight_relief"]:
            d.declare_partials("total_loads", "struct_weight_loads", rows=arange, cols=arange, val=1.0)

        if surface["distributed_fuel_weight"]:
            d.declare_partials("total_loads", "fuel_weight_loads", rows=arange, cols=arange, val=1.0)

        if "n_point_masses" in surface.keys():
            d.declare_partials("total_loads", "loads_from_point_masses", rows=arange, cols=arange, val=1.0)
            d.declare_partials("total_loads", "loads_from_thrusts", rows=arange, cols=arange, val=1.0)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total = np.array(inputs["loads"])

        if self.surface["struct_weight_relief"]:
            total += inputs["struct_weight_loads"]

        if self.surface["distributed_fuel_weight"]:
            total += inputs["fuel_weight_loads"]

        if "n_point_masses" in self.surface.keys():
            total += inputs["loads_from_point_masses"]
            total += inputs["loads_from_thrusts"]
        outputs["total_loads"] = total
