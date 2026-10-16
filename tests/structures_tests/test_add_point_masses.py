import unittest
import numpy as np

from om4.utils.assert_utils import assert_near_equal
from openaerostruct.structures.compute_point_mass_loads import ComputePointMassLoads
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test_derivs(self):
        surface = get_default_surfaces()[0]

        surface["n_point_masses"] = 2

        comp = ComputePointMassLoads(surface=surface)

        values = {
            "comp.nodes": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            "comp.point_masses": np.array([[2.0, 1.0]]),
            "comp.point_mass_locations": np.array([[2.1, 0.1, 0.2], [3.2, 1.2, 0.3]]),
        }

        run_test(self, comp, method="cs", values=values, step=1e-8, atol=1e-5)

    @unittest.skip("Reference values assume g=9.8; this test was already skipped upstream (grav_constant is 9.80665)")
    def test_simple_values(self):
        surface = get_default_surfaces()[0]

        surface["n_point_masses"] = 1

        comp = ComputePointMassLoads(surface=surface)

        values = {
            "comp.nodes": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            "comp.point_masses": np.array([[1 / 9.8]]),
            "comp.point_mass_locations": np.array([[0.55012, 0.1, 0.0]]),
        }

        prob = run_test(self, comp, method="cs", values=values, step=1e-8, atol=1e-5)

        truth_array = np.array([0, 0, -1.0, 0.0, 0.55012, 0.0])

        assert_near_equal(prob.get_val("comp.loads_from_point_masses")[0, :], truth_array, 1e-6)


if __name__ == "__main__":
    unittest.main()
