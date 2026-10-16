import unittest

import numpy as np

from openaerostruct.geometry.geometry_unification import GeomMultiUnification
from openaerostruct.geometry.utils import build_section_dicts
from openaerostruct.geometry.utils import stretch, sweep, dihedral
from openaerostruct.utils.testing import run_test, get_three_section_surface


class Test(unittest.TestCase):
    def test_no_shift(self):
        (surface, chord_bspline) = get_three_section_surface()
        sec_dicts = build_section_dicts(surface)

        comp = GeomMultiUnification(sections=sec_dicts, surface_name=surface["name"], shift_uni_mesh=False)

        values = {f"comp.sec{i}_def_mesh": sec_dicts[i]["mesh"] for i in range(3)}
        run_test(self, comp, method="cs", values=values)

    def test_shift(self):
        (surface, chord_bspline) = get_three_section_surface()

        # Apply some scalar mesh transformations
        surface["span"] = [5.0, 5.0, 3.0]
        surface["sweep"] = [-10.0, 10.0, -20.0]
        surface["dihedral"] = [-10.0, 10.0, -20.0]

        sec_dicts = build_section_dicts(surface)

        for i in range(surface["num_sections"]):
            sweep(sec_dicts[i]["mesh"], surface["sweep"][i], True)
            stretch(sec_dicts[i]["mesh"], surface["span"][i], True)
            dihedral(sec_dicts[i]["mesh"], surface["dihedral"][i], True)

        comp = GeomMultiUnification(sections=sec_dicts, surface_name=surface["name"], shift_uni_mesh=True)

        values = {f"comp.sec{i}_def_mesh": sec_dicts[i]["mesh"] for i in range(3)}
        run_test(self, comp, method="cs", values=values)

    def test_different_ny_with_t_over_c(self):
        (surface, chord_bspline) = get_three_section_surface()
        surface["ny"] = [5, 7, 9]
        sec_dicts = build_section_dicts(surface)

        comp = GeomMultiUnification(sections=sec_dicts, surface_name=surface["name"], shift_uni_mesh=True)

        values = {f"comp.sec{i}_def_mesh": sec_dicts[i]["mesh"] for i in range(3)}
        values.update(
            {f"comp.sec{i}_t_over_c": np.linspace(0.1, 0.2, sec_dicts[i]["mesh"].shape[1] - 1) for i in range(3)}
        )
        prob = run_test(self, comp, method="cs", values=values)
        expected = np.concatenate([values[f"comp.sec{i}_t_over_c"] for i in range(3)])
        np.testing.assert_allclose(prob.get_val("comp.surface_uni_t_over_c"), expected)


if __name__ == "__main__":
    unittest.main()
