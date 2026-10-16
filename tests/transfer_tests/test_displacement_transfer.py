import unittest
import numpy as np

from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]

        comp = DisplacementTransfer(surface=surface)

        mesh = surface["mesh"]
        nx, ny = mesh.shape[:2]

        rng = np.random.default_rng(0)
        values = {
            "comp.mesh": mesh,
            "comp.disp": rng.random((ny, 6)),
            "comp.transformation_matrix": rng.random((ny, 3, 3)),
            "comp.nodes": rng.random((ny, 3)),
        }

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
