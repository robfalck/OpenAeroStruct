import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import MultiSecGeometry
from openaerostruct.geometry.multi_section import build_section_dicts, multi_spline_join_kwargs, scalar_overlay, unify_mesh
from openaerostruct.utils.testing import get_two_section_surface


def _flow_values(prob):
    prob.set_val("v", 1.0, units="m/s")
    prob.set_val("alpha", 10.0, units="deg")
    prob.set_val("Mach_number", 0.3)
    prob.set_val("re", 1.0e5, units="1/m")
    prob.set_val("rho", 0.38, units="kg/m**3")
    prob.set_val("cg", np.zeros(3), units="m")


class Test(unittest.TestCase):
    def assert_check_results_sym(self, prob):
        """We put the assert_near_equal for the test_constraint and test construction cases here since they should both return the same results."""
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CD")[0], 0.02920058, 1e-3)
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CL")[0], 0.2999996, 1e-3)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.07335945, 1e-3)

    def test_constraint(self):
        surface, sec_chord_cp = get_two_section_surface()

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        subsystems = {
            "surface": MultiSecGeometry(
                surface=surface, joining_comp=True, dim_constr=[np.array([1, 0, 0]), np.array([1, 0, 0])]
            ),
            "aero_point_0": AeroPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
            ),
        }

        connections = [
            om.Connection(
                src="surface.surface_unification.surface_uni_mesh", tgt="aero_point_0.surface.def_mesh"
            ),
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
                    "alpha": DesignVar(lower=0.0, upper=10.0),
                },
                constraints={
                    # Equals=0.0 here is exactly rank-deficient: GeomMultiJoin's 2-row
                    # section_separation Jacobian has its rows exactly proportional for this
                    # (straight, untapered) two-section wing, and scipy's SLSQP chokes on an
                    # exactly-singular *equality*-constraint Jacobian ("Singular matrix C in
                    # LSQ subproblem"), even in OM3's own ScipyOptimizeDriver given bit-for-bit
                    # identical gradients -- confirmed by swapping OM3's original test to
                    # equals=0.0 and reproducing the identical failure there too. OM3's test
                    # only avoids this by declaring lower=0, upper=0 instead of equals=0, which
                    # happens to route through OpenMDAO's inequality code path. om4's Constraint
                    # has no such escape hatch (lb==ub always becomes a true equality), so a
                    # tight tolerance band is used instead of exact equality.
                    "surface.surface_joining.section_separation": Constraint(lower=-1e-6, upper=1e-6),
                    "aero_point_0.CL": Constraint(equals=0.3),
                    "aero_point_0.total_perf.S_ref_total": Constraint(equals=2.0),
                },
                objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-7, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): CM isn't part of any declared
        # design_var/constraint/objective here, so it can be left stale by run_optimizer();
        # force a full re-propagation before reading it.
        prob.run_model()

        self.assert_check_results_sym(prob)

    def test_construction(self):
        surface, sec_chord_cp = get_two_section_surface()

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        chord_kwargs = multi_spline_join_kwargs("chord_cp", section_surfaces, sec_chord_cp, surface["name"])

        subsystems = {
            **chord_kwargs["subsystems"],
            "surface": MultiSecGeometry(surface=surface),
            "aero_point_0": AeroPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
            ),
        }

        connections = [
            *chord_kwargs["connections"],
            om.Connection(
                src="surface.surface_unification.surface_uni_mesh", tgt="aero_point_0.surface.def_mesh"
            ),
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
                    "chord_cp_spline": DesignVar(lower=0.1, upper=10.0),
                    "alpha": DesignVar(lower=0.0, upper=10.0),
                },
                constraints={
                    "aero_point_0.CL": Constraint(equals=0.3),
                    "aero_point_0.total_perf.S_ref_total": Constraint(equals=2.0),
                },
                objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-7, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_constraint above.
        prob.run_model()

        self.assert_check_results_sym(prob)

    def test_asymmetrical(self):
        surface, sec_chord_cp = get_two_section_surface(sym=False, visc=False)

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        subsystems = {
            "surface": MultiSecGeometry(
                surface=surface, joining_comp=True, dim_constr=[np.array([1, 0, 0]), np.array([1, 0, 0])]
            ),
            "aero_point_0": AeroPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
            ),
        }

        connections = [
            om.Connection(
                src="surface.surface_unification.surface_uni_mesh", tgt="aero_point_0.surface.def_mesh"
            ),
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
                    "alpha": DesignVar(lower=0.0, upper=10.0),
                },
                constraints={
                    # Tolerance band instead of exact equality: see test_constraint above
                    # for why equals=0.0 makes GeomMultiJoin's section_separation constraint
                    # exactly rank-deficient and crashes SLSQP.
                    "surface.surface_joining.section_separation": Constraint(lower=-1e-6, upper=1e-6),
                    "aero_point_0.CL": Constraint(equals=0.3),
                    "aero_point_0.total_perf.S_ref_total": Constraint(equals=2.0),
                },
                objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-3, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_constraint above.
        prob.run_model()

        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CD")[0], 0.02902683, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CL")[0], 0.29999999, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.07375071, 1e-6)

    def test_visc(self):
        surface, sec_chord_cp = get_two_section_surface(sym=True, visc=True)

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        chord_kwargs = multi_spline_join_kwargs("chord_cp", section_surfaces, sec_chord_cp, surface["name"])

        subsystems = {
            **chord_kwargs["subsystems"],
            "surface": MultiSecGeometry(surface=surface),
            "aero_point_0": AeroPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"],
            ),
        }

        connections = [
            *chord_kwargs["connections"],
            om.Connection(
                src="surface.surface_unification.surface_uni_mesh", tgt="aero_point_0.surface.def_mesh"
            ),
            om.Connection(
                src="surface.surface_unification.surface_uni_mesh",
                tgt="aero_point_0.aero_states.surface_def_mesh",
            ),
            om.Connection(
                src="surface.surface_unification.surface_uni_t_over_c",
                tgt="aero_point_0.surface_perf.t_over_c",
            ),
        ]

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "chord_cp_spline": DesignVar(lower=0.1, upper=10.0),
                    "alpha": DesignVar(lower=0.0, upper=10.0),
                },
                constraints={
                    "aero_point_0.CL": Constraint(equals=0.3),
                    "aero_point_0.total_perf.S_ref_total": Constraint(equals=2.0),
                },
                objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-7, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_constraint above.
        prob.run_model()

        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CD")[0], 0.04946752, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.surface_perf.CL")[0], 0.3, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.05045758, 1e-6)


if __name__ == "__main__":
    unittest.main()
