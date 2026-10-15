import unittest
import numpy as np

from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        wing_dict = {"name": "wing", "mesh": np.zeros((2, 7)), "symmetry": True}
        tail_dict = {"name": "tail", "mesh": np.zeros((3, 5)), "symmetry": False}

        surfaces = [wing_dict, tail_dict]

        comp = MomentCoefficient(surfaces=surfaces)

        run_test(self, comp, complex_flag=True, method="cs")

    def test2(self):
        surfaces = get_default_surfaces()

        comp = MomentCoefficient(surfaces=surfaces)

        values = {"comp.S_ref_total": 1e4, "comp.cg": np.array([-10.0, 10.0, -10.0])}

        run_test(self, comp, complex_flag=True, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
