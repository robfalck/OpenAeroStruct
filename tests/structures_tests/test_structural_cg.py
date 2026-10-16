import unittest
import numpy as np

from openaerostruct.structures.structural_cg import StructuralCG
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]
        ny = surface["mesh"].shape[1]

        comp = StructuralCG(surface=surface)

        rng = np.random.default_rng(0)
        values = {
            "comp.nodes": rng.random((ny, 3)),
            "comp.structural_mass": 1.0,
            "comp.element_mass": np.ones((ny - 1)),
        }

        run_test(self, comp, values=values)


if __name__ == "__main__":
    unittest.main()
