import unittest

import numpy as np

from openaerostruct.common.reynolds_comp import ReynoldsComp
from openaerostruct.utils.testing import run_test


class Test(unittest.TestCase):
    def test_reynolds_derivs(self):
        comp = ReynoldsComp()

        rng = np.random.default_rng(0)
        values = {"comp.rho": rng.random(), "comp.mu": rng.random(), "comp.v": rng.random()}

        run_test(self, comp, method="cs", atol=1e-10, rtol=1e-10, values=values)


if __name__ == "__main__":
    unittest.main()
