import numpy as np
import unittest

from openaerostruct.geometry.geometry_multi_join import GeomMultiJoin
from openaerostruct.geometry.utils import build_section_dicts
from openaerostruct.utils.testing import run_test, get_three_section_surface


class Test(unittest.TestCase):
    def test(self):
        (surface, chord_bspline) = get_three_section_surface()
        sec_dicts = build_section_dicts(surface)

        comp = GeomMultiJoin(sections=sec_dicts, dim_constr=[np.ones(3), np.ones(3), np.ones(3)])

        values = {f"comp.sec{i}_join_mesh": sec_dicts[i]["mesh"] for i in range(3)}
        run_test(self, comp, method="cs", values=values)


if __name__ == "__main__":
    unittest.main()
