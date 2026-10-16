import unittest
import numpy as np

from openaerostruct.structures.weight import Weight
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]
        ny = surface["mesh"].shape[1]

        rng = np.random.default_rng(0)
        comp = Weight(surface=surface)

        run_test(self, comp, values={"comp.nodes": rng.random((ny, 3))})


if __name__ == "__main__":
    unittest.main()
