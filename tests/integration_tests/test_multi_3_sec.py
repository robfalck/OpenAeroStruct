import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import MultiSecGeometry
from openaerostruct.geometry.multi_section import build_section_dicts, scalar_overlay, unify_mesh
from openaerostruct.utils.testing import get_three_section_surface


def _flow_values(prob):
    prob.set_val("v", 1.0, units="m/s")
    prob.set_val("alpha", 10.0, units="deg")
    prob.set_val("Mach_number", 0.3)
    prob.set_val("re", 1.0e5, units="1/m")
    prob.set_val("rho", 0.38, units="kg/m**3")
    prob.set_val("cg", np.zeros(3), units="m")


def _build(surface, Sref):
    section_surfaces = build_section_dicts(surface)
    uni_mesh = unify_mesh(section_surfaces)
    surface["mesh"] = uni_mesh

    subsystems = {
        "surface": MultiSecGeometry(
            surface=surface,
            joining_comp=True,
            dim_constr=[np.array([1, 0, 0]), np.array([1, 0, 0]), np.array([1, 0, 0])],
        ),
        "aero_point_0": AeroPoint(
            surfaces=[scalar_overlay(surface)],
            promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
        ),
    }

    connections = [
        om.Connection(src="surface.surface_unification.surface_uni_mesh", tgt="aero_point_0.surface.def_mesh"),
        om.Connection(
            src="surface.surface_unification.surface_uni_mesh",
            tgt="aero_point_0.aero_states.surface_def_mesh",
        ),
    ]

    prob = om.Problem(
        model=om.Group(
            subsystems=subsystems,
            connections=connections,
            design_vars={
                "surface.sec0.chord_cp": DesignVar(lower=0.1, upper=10.0),
                "surface.sec1.chord_cp": DesignVar(lower=0.1, upper=10.0),
                "surface.sec2.chord_cp": DesignVar(lower=0.1, upper=10.0),
                "alpha": DesignVar(lower=0.0, upper=10.0),
            },
            constraints={
                # Tolerance band instead of exact equality: GeomMultiJoin's section_separation
                # Jacobian is exactly rank-deficient for this straight, untapered wing, which
                # crashes scipy's SLSQP ("Singular matrix C in LSQ subproblem") when declared as
                # a true equality -- same issue documented in test_multi_2_sec.py.
                "surface.surface_joining.section_separation": Constraint(lower=-1e-6, upper=1e-6),
                "aero_point_0.CL": Constraint(equals=0.3),
                "aero_point_0.total_perf.S_ref_total": Constraint(equals=Sref),
            },
            objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
            optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-7, maxiter=1000),
        )
    )

    _flow_values(prob)
    return prob


class Test(unittest.TestCase):
    def test_symmetric(self):
        surface, sec_chord_cp = get_three_section_surface(sym=True, visc=False)
        prob = _build(surface, Sref=2.0)

        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): CM isn't part of any declared
        # design_var/constraint/objective here, so it can be left stale by run_optimizer();
        # force a full re-propagation before reading it.
        prob.run_model()

        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CD")[0], 0.02126492, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CL")[0], 0.3, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.10778177, 1e-6)

    def test_asymmetric(self):
        surface, sec_chord_cp = get_three_section_surface(sym=False, visc=False)
        prob = _build(surface, Sref=2.5)

        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_symmetric above.
        prob.run_model()

        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CD")[0], 0.02270839, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CL")[0], 0.3, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.08717908, 1e-6)


if __name__ == "__main__":
    unittest.main()
