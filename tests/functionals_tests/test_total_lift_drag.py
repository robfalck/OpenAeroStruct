import unittest

import numpy as np

from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        wing_dict = {"name": "wing", "mesh": np.zeros((2, 7, 3)), "symmetry": True}
        tail_dict = {"name": "tail", "mesh": np.zeros((3, 5, 3)), "symmetry": False}

        surfaces = [wing_dict, tail_dict]

        comp = TotalLiftDrag(surfaces=surfaces)

        run_test(self, comp)

    # This is known to have some issues for sufficiently small values of S_ref_total
    # There is probably a derivative bug somewhere in the moment_coefficient.py calcs
    def test2(self):
        surfaces = get_default_surfaces()

        comp = TotalLiftDrag(surfaces=surfaces)

        run_test(self, comp, values={"comp.S_ref_total": 10.0})


if __name__ == "__main__":
    unittest.main()
