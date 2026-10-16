import unittest
import numpy as np

from openaerostruct.transfer.compute_transformation_matrix import ComputeTransformationMatrix
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]

        comp = ComputeTransformationMatrix(surface=surface)

        ny = surface["mesh"].shape[1]
        rng = np.random.default_rng(0)
        values = {"comp.disp": rng.random((ny, 6)) * 100.0}

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
