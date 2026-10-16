import unittest
import numpy as np

from openaerostruct.structures.vonmises_tube import VonMisesTube
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]

        comp = VonMisesTube(surface=surface)

        ny = surface["mesh"].shape[1]

        #  define the spar with y out the wing
        nodes = np.zeros((ny, 3))
        nodes[:, 0] = np.linspace(0, 0.01, ny)
        nodes[:, 1] = np.linspace(0, 1, ny)

        radius = 0.01 * np.ones((ny - 1))

        disp = np.zeros((ny, 6))
        for i in range(6):
            disp[:, i] = np.linspace(0, 0.001, ny)

        values = {"comp.nodes": nodes, "comp.radius": radius, "comp.disp": disp}

        run_test(self, comp, method="cs", values=values, atol=2e-4, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
