import unittest

from openaerostruct.aerodynamics.pg_scale import ScaleFromPrandtlGlauert, ScaleToPrandtlGlauert
from openaerostruct.aerodynamics.pg_wind_rotation import RotateFromWindFrame, RotateToWindFrame
from openaerostruct.aerodynamics.pg_transform import PGTransform, InversePGTransform
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test_to_wind_derivs(self):
        surfaces = get_default_surfaces()

        comp = RotateToWindFrame(surfaces=surfaces, rotational=True)

        run_test(self, comp)

    def test_from_wind_derivs(self):
        surfaces = get_default_surfaces()

        comp = RotateFromWindFrame(surfaces=surfaces)

        run_test(self, comp)

    def test_scale_to_pg(self):
        surfaces = get_default_surfaces()

        comp = ScaleToPrandtlGlauert(surfaces=surfaces, rotational=True)

        run_test(self, comp)

    def test_scale_from_pg(self):
        surfaces = get_default_surfaces()

        comp = ScaleFromPrandtlGlauert(surfaces=surfaces)

        run_test(self, comp)

    def test_pg_transform(self):
        surfaces = get_default_surfaces()

        comp = PGTransform(surfaces=surfaces, rotational=True)

        run_test(self, comp)

    def test_inverse_pg_transform(self):
        surfaces = get_default_surfaces()

        comp = InversePGTransform(surfaces=surfaces)

        run_test(self, comp)


if __name__ == "__main__":
    unittest.main()
