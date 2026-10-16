import unittest
import numpy as np

from openaerostruct.structures.vonmises_wingbox import VonMisesWingbox
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]

        # turn down some of these properties, so the absolute deriv error isn't magnified
        surface["E"] = 7
        surface["G"] = 3
        surface["yield"] = 0.02

        surface["strength_factor_for_upper_skin"] = 1.0

        comp = VonMisesWingbox(surface=surface)

        ny = surface["mesh"].shape[1]

        values = {
            "comp.nodes": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            "comp.disp": np.ones((ny, 6)),
        }
        for name in ("Qz", "J", "A_enc", "spar_thickness", "htop", "hbottom", "hfront", "hrear"):
            values["comp." + name] = np.ones((ny - 1))

        run_test(self, comp, values=values, step=1e-8, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
