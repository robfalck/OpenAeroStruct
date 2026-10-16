import unittest
import numpy as np

import om4.api as om
from om4.utils.assert_utils import assert_near_equal
from openaerostruct.structures.compute_thrust_loads import ComputeThrustLoads
from openaerostruct.utils.testing import run_test, get_default_surfaces


derivs_added = False


class Test(unittest.TestCase):
    def test_no_derivs(self):
        surface = get_default_surfaces()[0]

        surface["n_point_masses"] = 1

        comp = ComputeThrustLoads(surface=surface)

        nodesval = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])

        engine_thrusts = np.array([[10.0]])

        point_mass_locations = np.array([[0.0, 0.1, -0.5]])

        prob = om.Problem(model=om.Group(subsystems={"comp": comp}))
        prob.set_val("comp.nodes", nodesval)
        prob.set_val("comp.engine_thrusts", engine_thrusts)
        prob.set_val("comp.point_mass_locations", point_mass_locations)
        prob.run_model()

        truth_array = np.array([-10.0, 0, 0.0, 0.0, 5.0, 1.0])

        assert_near_equal(prob.get_val("comp.loads_from_thrusts")[0, :], truth_array, 1e-6)

    @unittest.skipUnless(derivs_added, "Analytic derivs not added yet")
    def test_derivs(self):
        surface = get_default_surfaces()[0]

        surface["n_point_masses"] = 2

        comp = ComputeThrustLoads(surface=surface)

        values = {
            "comp.nodes": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            "comp.engine_thrusts": np.array([[2.0, 1.0]]),
            "comp.point_mass_locations": np.array([[2.1, 0.1, 0.2], [3.2, 1.2, 0.3]]),
        }

        run_test(self, comp, values=values, step=1e-8, atol=1e-5)

    @unittest.skipUnless(derivs_added, "Analytic derivs not added yet")
    def test_simple_values(self):
        surface = get_default_surfaces()[0]

        surface["n_point_masses"] = 1

        comp = ComputeThrustLoads(surface=surface)

        values = {
            "comp.nodes": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            "comp.engine_thrusts": np.array([[1 / 9.8]]),
            "comp.point_mass_locations": np.array([[0.55012, 0.1, 0.0]]),
        }

        prob = run_test(self, comp, values=values, step=1e-8, atol=1e-5)

        truth_array = np.array([0, 0, -1.0, 0.0, 0.55012, 0.0])

        assert_near_equal(prob.get_val("comp.loads_from_thrusts")[0, :], truth_array, 1e-6)


if __name__ == "__main__":
    unittest.main()
