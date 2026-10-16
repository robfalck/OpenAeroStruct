import unittest

import numpy as np

from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test_0(self):
        surface = get_default_surfaces()[0]

        comp = FuelLoads(surface=surface)

        nodesval = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])

        run_test(self, comp, values={"comp.nodes": nodesval}, atol=1e-2, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
