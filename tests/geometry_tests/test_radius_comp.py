import unittest

import numpy as np

from openaerostruct.geometry.radius_comp import RadiusComp
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = RadiusComp(surface=surfaces[0])
        ny = surfaces[0]["mesh"].shape[1]

        run_test(
            self, comp, values={"comp.mesh": surfaces[0]["mesh"], "comp.t_over_c": np.linspace(0.1, 0.5, num=ny - 1)}
        )


if __name__ == "__main__":
    unittest.main()
