import unittest

import numpy as np

from openaerostruct.aerodynamics.rotational_velocity import RotationalVelocity
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = RotationalVelocity(surfaces=surfaces)

        run_test(self, comp)

    def test_rotation_option_derivatives(self):
        surfaces = get_default_surfaces()

        comp = RotationalVelocity(surfaces=surfaces)

        rng = np.random.default_rng(0)
        values = {
            "comp.omega": np.array([0.3, 0.4, -0.1]),
            "comp.cg": np.array([0.1, 0.6, 0.4]),
            "comp.coll_pts": rng.random(comp.inputs["coll_pts"].val.shape),
        }

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
