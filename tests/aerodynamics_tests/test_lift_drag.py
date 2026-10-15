import unittest

import numpy as np

from openaerostruct.aerodynamics.lift_drag import LiftDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = LiftDrag(surface=surfaces[0])

        run_test(self, comp)

    def test_derivs_with_sideslip(self):
        surfaces = get_default_surfaces()

        # Use Tail since it is not symmetric.
        comp = LiftDrag(surface=surfaces[1])

        rng = np.random.default_rng(0)
        values = {
            "comp.alpha": 3.0,
            "comp.beta": 15.0,
            "comp.sec_forces": 10.0 * rng.random(comp.inputs["sec_forces"].val.shape),
        }

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
