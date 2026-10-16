import unittest
import numpy as np

from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]

        comp = LoadTransfer(surface=surface)

        nx = surface["mesh"].shape[0]
        ny = surface["mesh"].shape[1]

        rng = np.random.default_rng(0)
        values = {
            "comp.def_mesh": rng.random((nx, ny, 3)),
            "comp.sec_forces": rng.random((nx - 1, ny - 1, 3)),
        }

        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
