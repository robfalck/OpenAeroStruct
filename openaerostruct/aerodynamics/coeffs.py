from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl


class Coeffs(om.ExplicitComponent):
    """Compute lift and drag coefficients for each individual lifting surface.

    Parameters
    ----------
    S_ref : float
        The reference areas of the lifting surface.
    L : float
        Total lift for the lifting surface.
    D : float
        Total drag for the lifting surface.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    CL1 : float
        Induced coefficient of lift (CL) for the lifting surface.
    CDi : float
        Induced coefficient of drag (CD) for the lifting surface.
    """

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        d.add_input("S_ref", val=1.0, units="m**2", tags=["mphys_coupling"])
        d.add_input("L", val=1.0, units="N")
        d.add_input("D", val=1.0, units="N")
        d.add_input("v", val=1.0, units="m/s", tags=["mphys_input"])
        d.add_input("rho", val=1.0, units="kg/m**3", tags=["mphys_input"])

        d.add_output("CL1", val=0.0)
        d.add_output("CDi", val=0.0)

        d.declare_partials("CL1", "L")
        d.declare_partials("CDi", "D")
        d.declare_partials("CL1", "v")
        d.declare_partials("CDi", "v")
        d.declare_partials("CL1", "rho")
        d.declare_partials("CDi", "rho")
        d.declare_partials("CL1", "S_ref")
        d.declare_partials("CDi", "S_ref")
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        S_ref = inputs["S_ref"]
        rho = inputs["rho"]
        v = inputs["v"]
        L = inputs["L"]
        D = inputs["D"]

        outputs["CL1"] = L / (0.5 * rho * v**2 * S_ref)
        outputs["CDi"] = D / (0.5 * rho * v**2 * S_ref)

    def compute_partials(self, inputs, partials):
        S_ref = inputs["S_ref"]
        rho = inputs["rho"]
        v = inputs["v"]
        L = inputs["L"]
        D = inputs["D"]

        partials["CL1", "L"] = 1.0 / (0.5 * rho * v**2 * S_ref)
        partials["CDi", "D"] = 1.0 / (0.5 * rho * v**2 * S_ref)

        partials["CL1", "v"] = -2.0 * L / (0.5 * rho * v**3 * S_ref)
        partials["CDi", "v"] = -2.0 * D / (0.5 * rho * v**3 * S_ref)

        partials["CL1", "rho"] = -L / (0.5 * rho**2 * v**2 * S_ref)
        partials["CDi", "rho"] = -D / (0.5 * rho**2 * v**2 * S_ref)

        partials["CL1", "S_ref"] = -L / (0.5 * rho * v**2 * S_ref**2)
        partials["CDi", "S_ref"] = -D / (0.5 * rho * v**2 * S_ref**2)
