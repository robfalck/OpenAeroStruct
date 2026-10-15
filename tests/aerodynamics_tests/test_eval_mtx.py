import unittest

import om4.api as om
from om4.derivatives.complex_step import ComplexStep

from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx
from openaerostruct.utils.testing import run_test, get_default_surfaces, get_ground_effect_surfaces


def _flip_to_right(surfaces):
    # flip each surface to lie on right
    for surface in surfaces:
        surface["mesh"] = surface["mesh"][:, ::-1, :]
        surface["mesh"][:, :, 1] *= -1.0


def _check_assembled_jac(comp):
    """Check that the partials are usable under a sparse-assembled DirectSolver.

    The tolerance is deliberately huge (as in OM3): this only verifies that the check runs.
    """
    model = om.Group(
        subsystems={"comp": comp},
        linear_solver=om.DirectSolver(assembled_format="sparse"),
    )
    prob = om.Problem(model=model)
    prob.run_model()

    data = prob.check_partials(differentiator=ComplexStep(), only_incorrect=True, atol=1e20, rtol=1e-6)
    assert not data


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        run_test(self, comp, complex_flag=True)

    def test_assembled_jac(self):
        surfaces = get_default_surfaces()

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        _check_assembled_jac(comp)


class GroundEffectTest(unittest.TestCase):
    def test(self):
        surfaces = get_ground_effect_surfaces()

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        run_test(self, comp, complex_flag=True)

    def test_assembled_jac(self):
        surfaces = get_ground_effect_surfaces()

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        _check_assembled_jac(comp)


class TestRightWing(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()
        _flip_to_right(surfaces)

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        run_test(self, comp, complex_flag=True)

    def test_assembled_jac(self):
        surfaces = get_default_surfaces()
        _flip_to_right(surfaces)

        comp = EvalVelMtx(surfaces=surfaces, num_eval_points=2, eval_name="test_name")

        _check_assembled_jac(comp)


if __name__ == "__main__":
    unittest.main()
