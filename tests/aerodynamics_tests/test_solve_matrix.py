import unittest
import numpy as np

from openaerostruct.aerodynamics.solve_matrix import SolveMatrix
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = SolveMatrix(surfaces=surfaces)

        system_size = 0
        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            system_size += (nx - 1) * (ny - 1)

        values = {"comp.rhs": np.ones((system_size)), "comp.mtx": np.identity((system_size))}

        run_test(self, comp, values=values)


if __name__ == "__main__":
    unittest.main()
