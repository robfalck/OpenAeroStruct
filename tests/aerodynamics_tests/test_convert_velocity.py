import unittest

import numpy as np

from openaerostruct.aerodynamics.convert_velocity import ConvertVelocity
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = ConvertVelocity(surfaces=surfaces)

        run_test(self, comp)

    def test_rotation_option_derivatives(self):
        surfaces = get_default_surfaces()

        comp = ConvertVelocity(surfaces=surfaces, rotational=True)

        rng = np.random.default_rng(0)
        values = {
            "comp.rotational_velocities": rng.random(comp.inputs["rotational_velocities"].val.shape),
            "comp.beta": 15.0,
        }

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
