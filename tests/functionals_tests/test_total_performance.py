import unittest

import numpy as np

import om4.api as om
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.utils.testing import get_default_surfaces


class Test(unittest.TestCase):
    def _problem(self, user_specified_Sref, internally_connect_fuelburn):
        surfaces = get_default_surfaces()
        group = TotalPerformance(
            surfaces=surfaces,
            user_specified_Sref=user_specified_Sref,
            internally_connect_fuelburn=internally_connect_fuelburn,
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )
        prob = om.Problem(model=om.Group(subsystems={"perf": group}))
        prob.set_val("wing_structural_mass", 1000.0, units="kg")
        prob.set_val("wing_CL", 0.5)
        prob.set_val("wing_CD", 0.03)
        prob.set_val("W0", 5000.0, units="kg")
        prob.set_val("v", 200.0, units="m/s")
        prob.set_val("rho", 0.4, units="kg/m**3")
        if user_specified_Sref:
            prob.set_val("S_ref_total", 400.0, units="m**2")
        prob.run_model()
        return prob

    def test_totals(self):
        for user_specified_Sref in (False, True):
            prob = self._problem(user_specified_Sref, True)
            totals = prob.check_totals(
                of=["CL", "CD", "fuelburn", "L_equals_W", "cg"],
                wrt=["wing_structural_mass", "W0", "v", "rho"],
                only_incorrect=True,
                atol=1e-5,
                rtol=1e-5,
            )
            assert not totals, totals

    def test_fuelburn_connected(self):
        prob = self._problem(False, True)
        assert prob.get_val("fuelburn")[0] > 0.0
        total_weight = prob.get_val("total_weight")[0]
        assert total_weight > (5000.0 + 1000.0) * 9.81

    def test_fuelburn_not_connected(self):
        prob = self._problem(False, False)
        np.testing.assert_allclose(prob.get_val("perf.L_equals_W.fuelburn"), 123.0)


if __name__ == "__main__":
    unittest.main()
