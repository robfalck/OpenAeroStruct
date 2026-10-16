from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class SectionPropertiesTube(om.ExplicitComponent):
    """
    Compute geometric properties for a tube element.
    The thicknesses are added to the interior of the element, so the
    'radius' value is the outer radius of the tube.

    Parameters
    ----------
    radius : numpy array
        Outer radii for each FEM element.
    thickness : numpy array
        Tube thickness for each FEM element.

    Returns
    -------
    A : numpy array
        Cross-sectional area for each FEM element.
    Iy : numpy array
        Area moment of inertia around the y-axis for each FEM element.
    Iz : numpy array
        Area moment of inertia around the z-axis for each FEM element.
    J : numpy array
        Polar moment of inertia for each FEM element.
    """

    surface: Surface
    check_partials_options: om.CheckPartialsOptions | None = om.CheckPartialsOptions(differentiator="cs")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("radius", val=np.ones((ny - 1)), units="m")
        d.add_input("thickness", val=np.ones((ny - 1)) * 0.1, units="m")
        d.add_output("A", val=np.zeros((ny - 1)), units="m**2")
        d.add_output("Iy", val=np.zeros((ny - 1)), units="m**4")
        d.add_output("Iz", val=np.zeros((ny - 1)), units="m**4")
        d.add_output("J", val=np.zeros((ny - 1)), units="m**4")

        a = np.arange((ny - 1))
        d.declare_partials("*", "*", rows=a, cols=a)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pi = np.pi

        # Add thickness to the interior of the radius.
        # The outer radius is the inputs['radius'] amount.
        r1 = inputs["radius"] - inputs["thickness"]
        r2 = inputs["radius"]

        # Compute the area, area moments of inertia, and polar moment of inertia
        outputs["A"] = pi * (r2**2 - r1**2)
        outputs["Iy"] = pi * (r2**4 - r1**4) / 4.0
        outputs["Iz"] = pi * (r2**4 - r1**4) / 4.0
        outputs["J"] = pi * (r2**4 - r1**4) / 2.0

    def compute_partials(self, inputs, partials):
        pi = np.pi
        radius = inputs["radius"]
        t = inputs["thickness"]
        r1 = radius - t
        r2 = radius

        dr1_dr = 1.0
        dr2_dr = 1.0
        dr1_dt = -1.0
        dr2_dt = 0.0

        r1_3 = r1**3
        r2_3 = r2**3

        partials["A", "radius"] = 2 * pi * (r2 * dr2_dr - r1 * dr1_dr)
        partials["A", "thickness"] = 2 * pi * (r2 * dr2_dt - r1 * dr1_dt)
        partials["Iy", "radius"] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        partials["Iy", "thickness"] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        partials["Iz", "radius"] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        partials["Iz", "thickness"] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        partials["J", "radius"] = 2 * pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        partials["J", "thickness"] = 2 * pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
