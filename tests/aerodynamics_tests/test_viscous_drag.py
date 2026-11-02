import unittest

import numpy as np

import om4.api as om

from openaerostruct.aerodynamics.viscous_drag import ViscousDrag
from openaerostruct.utils.om4_utils import spline_or_broadcast
from openaerostruct.utils.testing import run_test, get_default_surfaces


def _viscous_group(surface):
    ny = surface["mesh"].shape[1]
    n_cp = len(surface["t_over_c_cp"])

    x_interp = np.linspace(0.0, 1.0, int(ny - 1))
    return om.Group(
        subsystems={
            "t_over_c_bsp": spline_or_broadcast(
                n_cp=n_cp,
                x_interp=x_interp,
                control_name="t_over_c_cp",
                interp_name="t_over_c",
                units=None,
                control_val=np.asarray(surface["t_over_c_cp"], dtype=float),
                promotes_inputs=["t_over_c_cp"],
                promotes_outputs=["t_over_c"],
            ),
            "viscousdrag": ViscousDrag(surface=surface, with_viscous=True, promotes_inputs=["t_over_c"]),
        }
    )


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]
        surface["t_over_c_cp"] = np.array([0.1, 0.15, 0.2])

        run_test(self, _viscous_group(surface), complex_flag=True)

    def test_2(self):
        surface = get_default_surfaces()[0]
        surface["k_lam"] = 0.5

        run_test(self, _viscous_group(surface), complex_flag=True)


if __name__ == "__main__":
    unittest.main()
