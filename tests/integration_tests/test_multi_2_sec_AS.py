"""We verify the results of the two-section aerostructural optimization using both the construction and constraint
methods of joining sections. The result should be the same so we compare with the same reference value. We also perform
the same optimization for an asymmetical two-section mesh which is compared against its own reference value.
"""

import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.integration.aerostruct_groups import AerostructPoint, MultiSecAerostructGeometry
from openaerostruct.geometry.multi_section import build_section_dicts, multi_spline_join_kwargs, scalar_overlay, unify_mesh
from openaerostruct.utils.constants import grav_constant
from openaerostruct.utils.testing import get_two_section_surface_AS


def _flow_values(prob):
    prob.set_val("v", 248.136, units="m/s")
    prob.set_val("alpha", 9.0, units="deg")
    prob.set_val("Mach_number", 0.84)
    prob.set_val("re", 1.0e6, units="1/m")
    prob.set_val("rho", 0.38, units="kg/m**3")
    prob.set_val("CT", grav_constant * 17.0e-6, units="1/s")
    prob.set_val("R", 11.165e6, units="m")
    prob.set_val("W0", 0.4 * 3e5, units="kg")
    prob.set_val("speed_of_sound", 295.4, units="m/s")
    prob.set_val("load_factor", 1.0)
    prob.set_val("empty_cg", np.zeros(3), units="m")


def _aero_struct_connections(name, point_name):
    com_name = point_name + "." + name + "_perf"
    return [
        om.Connection(
            src=name + ".local_stiff_transformed", tgt=point_name + ".coupled." + name + ".local_stiff_transformed"
        ),
        om.Connection(src=name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"),
        om.Connection(src=name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"),
        om.Connection(src=name + ".radius", tgt=com_name + ".radius"),
        om.Connection(src=name + ".thickness", tgt=com_name + ".thickness"),
        om.Connection(src=name + ".nodes", tgt=com_name + ".nodes"),
        om.Connection(src=name + ".cg_location", tgt=point_name + "." + "total_perf." + name + "_cg_location"),
        om.Connection(
            src=name + ".structural_mass", tgt=point_name + "." + "total_perf." + name + "_structural_mass"
        ),
        om.Connection(src=name + ".t_over_c", tgt=com_name + ".t_over_c"),
    ]


class Test(unittest.TestCase):
    def assert_check_results_sym(self, prob):
        """We put the assert_near_equal for the test_constraint and test construction cases here since they should both return the same results."""
        assert_near_equal(prob.get_val("AS_point_0.fuelburn")[0], 109110.19599915, 1e-3)

    def test_constraint(self):
        surface, sec_twist_cp = get_two_section_surface_AS()

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        name = surface["name"]
        point_name = "AS_point_0"

        subsystems = {
            name: MultiSecAerostructGeometry(
                surface=surface,
                joining_comp=True,
                dim_constr=[np.array([1, 0, 0]), np.array([1, 0, 0])],
                shift_uni_mesh=False,
            ),
            point_name: AerostructPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=[
                    "v",
                    "alpha",
                    "Mach_number",
                    "re",
                    "rho",
                    "CT",
                    "R",
                    "W0",
                    "speed_of_sound",
                    "empty_cg",
                    "load_factor",
                ],
            ),
        }

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=_aero_struct_connections(name, point_name),
                design_vars={
                    "alpha": DesignVar(lower=-10.0, upper=10.0),
                    "surface.sec0.twist_cp": DesignVar(lower=-10.0, upper=15.0),
                    "surface.sec1.twist_cp": DesignVar(lower=-10.0, upper=15.0),
                    "surface.thickness_cp": DesignVar(
                        lower=0.01, upper=0.5, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    ),
                },
                constraints={
                    "AS_point_0.L_equals_W": Constraint(equals=0.0),
                    "AS_point_0.surface_perf.failure": Constraint(upper=0.0),
                    "AS_point_0.surface_perf.thickness_intersects": Constraint(upper=0.0),
                    # Tolerance band instead of exact equality: see test_multi_2_sec.py for why
                    # GeomMultiJoin's section_separation Jacobian is exactly rank-deficient for
                    # this wing, which crashes scipy's SLSQP when declared as a true equality.
                    "surface.surface_joining.section_separation": Constraint(lower=-1e-6, upper=1e-6),
                },
                objectives={"AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=5e-1))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-10, maxiter=1000),
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
        surface, sec_twist_cp = get_two_section_surface_AS()

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        twist_kwargs = multi_spline_join_kwargs("twist_cp", section_surfaces, sec_twist_cp, surface["name"], units="deg")

        name = surface["name"]
        point_name = "AS_point_0"

        subsystems = {
            **twist_kwargs["subsystems"],
            name: MultiSecAerostructGeometry(surface=surface),
            point_name: AerostructPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=[
                    "v",
                    "alpha",
                    "Mach_number",
                    "re",
                    "rho",
                    "CT",
                    "R",
                    "W0",
                    "speed_of_sound",
                    "empty_cg",
                    "load_factor",
                ],
            ),
        }

        connections = [*twist_kwargs["connections"], *_aero_struct_connections(name, point_name)]

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "alpha": DesignVar(lower=-10.0, upper=10.0),
                    "twist_cp_spline": DesignVar(lower=-10.0, upper=15.0),
                    "surface.thickness_cp": DesignVar(
                        lower=0.01, upper=0.5, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    ),
                },
                constraints={
                    "AS_point_0.L_equals_W": Constraint(equals=0.0),
                    "AS_point_0.surface_perf.failure": Constraint(upper=0.0),
                    "AS_point_0.surface_perf.thickness_intersects": Constraint(upper=0.0),
                },
                objectives={"AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-10, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_constraint above.
        prob.run_model()

        self.assert_check_results_sym(prob)

    def test_asymmetrical(self):
        surface, sec_twist_cp = get_two_section_surface_AS(sym=False)

        section_surfaces = build_section_dicts(surface)
        uni_mesh = unify_mesh(section_surfaces)
        surface["mesh"] = uni_mesh

        twist_kwargs = multi_spline_join_kwargs("twist_cp", section_surfaces, sec_twist_cp, surface["name"], units="deg")

        name = surface["name"]
        point_name = "AS_point_0"

        subsystems = {
            **twist_kwargs["subsystems"],
            name: MultiSecAerostructGeometry(surface=surface),
            point_name: AerostructPoint(
                surfaces=[scalar_overlay(surface)],
                promotes_inputs=[
                    "v",
                    "alpha",
                    "Mach_number",
                    "re",
                    "rho",
                    "CT",
                    "R",
                    "W0",
                    "speed_of_sound",
                    "empty_cg",
                    "load_factor",
                ],
            ),
        }

        connections = [*twist_kwargs["connections"], *_aero_struct_connections(name, point_name)]

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "alpha": DesignVar(lower=-10.0, upper=10.0),
                    "twist_cp_spline": DesignVar(lower=-10.0, upper=15.0),
                    "surface.thickness_cp": DesignVar(
                        lower=0.01, upper=0.5, transform=AffineTransform.from_scaler_adder(scaler=1e2)
                    ),
                },
                constraints={
                    "AS_point_0.L_equals_W": Constraint(equals=0.0),
                    "AS_point_0.surface_perf.failure": Constraint(upper=0.0),
                    "AS_point_0.surface_perf.thickness_intersects": Constraint(upper=0.0),
                },
                objectives={"AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-10, maxiter=1000),
            )
        )

        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see test_constraint above.
        prob.run_model()

        assert_near_equal(prob.get_val("AS_point_0.fuelburn")[0], 85719.42698562, 1e-3)


if __name__ == "__main__":
    unittest.main()
