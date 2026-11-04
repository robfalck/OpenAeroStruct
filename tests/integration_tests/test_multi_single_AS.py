"""We verify the results of the two-section aerostructural analysis to the same wing analysed with single section
geometry. We verify the results are the same.
"""

import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint, MultiSecAerostructGeometry
from openaerostruct.geometry.multi_section import build_section_dicts, multi_spline_join_kwargs, scalar_overlay, unify_mesh
from openaerostruct.utils.constants import grav_constant
from openaerostruct.utils.testing import get_single_section_surface_AS, get_two_section_surface_AS


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


class Test(unittest.TestCase):
    def test(self):
        # Single-section aerostructural optimization
        surfaceSingle = get_single_section_surface_AS()

        name0 = surfaceSingle["name"]
        point_name0 = "AS_point_0"

        aerostruct_geom_group = AerostructGeometry(surface=surfaceSingle)
        AS_point_0 = AerostructPoint(
            surfaces=[surfaceSingle],
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
        )

        com_name0 = point_name0 + "." + name0 + "_perf"
        connections0 = [
            om.Connection(
                src=name0 + ".local_stiff_transformed",
                tgt=point_name0 + ".coupled." + name0 + ".local_stiff_transformed",
            ),
            om.Connection(src=name0 + ".nodes", tgt=point_name0 + ".coupled." + name0 + ".nodes"),
            om.Connection(src=name0 + ".mesh", tgt=point_name0 + ".coupled." + name0 + ".mesh"),
            om.Connection(src=name0 + ".radius", tgt=com_name0 + ".radius"),
            om.Connection(src=name0 + ".thickness", tgt=com_name0 + ".thickness"),
            om.Connection(src=name0 + ".nodes", tgt=com_name0 + ".nodes"),
            om.Connection(
                src=name0 + ".cg_location", tgt=point_name0 + "." + "total_perf." + name0 + "_cg_location"
            ),
            om.Connection(
                src=name0 + ".structural_mass",
                tgt=point_name0 + "." + "total_perf." + name0 + "_structural_mass",
            ),
            om.Connection(src=name0 + ".t_over_c", tgt=com_name0 + ".t_over_c"),
        ]

        prob = om.Problem(
            model=om.Group(
                subsystems={name0: aerostruct_geom_group, point_name0: AS_point_0},
                connections=connections0,
                design_vars={"alpha": DesignVar(lower=-20.0, upper=20.0)},
                constraints={"AS_point_0.L_equals_W": Constraint(equals=0.0)},
                objectives={"AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-6, maxiter=1000),
            )
        )
        _flow_values(prob)
        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): CM isn't part of any declared
        # design_var/constraint/objective here, so it can be left stale by run_optimizer();
        # force a full re-propagation before reading it.
        prob.run_model()

        # Two-section aerostructural optimization, joined by construction (shared twist_cp)
        surfaceMulti, sec_twist_cp = get_two_section_surface_AS()
        surfaceMulti["name"] = "surfaceMulti"

        section_surfaces = build_section_dicts(surfaceMulti)
        uni_mesh = unify_mesh(section_surfaces)
        surfaceMulti["mesh"] = uni_mesh

        twist_kwargs = multi_spline_join_kwargs(
            "twist_cp", section_surfaces, sec_twist_cp, surfaceMulti["name"], units="deg"
        )

        name1 = surfaceMulti["name"]
        point_name1 = "AS_point_1"

        multi_geom_group = MultiSecAerostructGeometry(surface=surfaceMulti)
        AS_point_1 = AerostructPoint(
            surfaces=[scalar_overlay(surfaceMulti)],
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
        )

        com_name1 = point_name1 + "." + name1 + "_perf"
        connections1 = [
            *twist_kwargs["connections"],
            om.Connection(
                src=name1 + ".local_stiff_transformed",
                tgt=point_name1 + ".coupled." + name1 + ".local_stiff_transformed",
            ),
            om.Connection(src=name1 + ".nodes", tgt=point_name1 + ".coupled." + name1 + ".nodes"),
            om.Connection(src=name1 + ".mesh", tgt=point_name1 + ".coupled." + name1 + ".mesh"),
            om.Connection(src=name1 + ".radius", tgt=com_name1 + ".radius"),
            om.Connection(src=name1 + ".thickness", tgt=com_name1 + ".thickness"),
            om.Connection(src=name1 + ".nodes", tgt=com_name1 + ".nodes"),
            om.Connection(
                src=name1 + ".cg_location", tgt=point_name1 + "." + "total_perf." + name1 + "_cg_location"
            ),
            om.Connection(
                src=name1 + ".structural_mass",
                tgt=point_name1 + "." + "total_perf." + name1 + "_structural_mass",
            ),
            om.Connection(src=name1 + ".t_over_c", tgt=com_name1 + ".t_over_c"),
        ]

        prob2 = om.Problem(
            model=om.Group(
                subsystems={**twist_kwargs["subsystems"], name1: multi_geom_group, point_name1: AS_point_1},
                connections=connections1,
                design_vars={"alpha": DesignVar(lower=-20.0, upper=20.0)},
                constraints={"AS_point_1.L_equals_W": Constraint(equals=0.0)},
                objectives={"AS_point_1.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-6, maxiter=1000),
            )
        )
        _flow_values(prob2)
        prob2.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): see above.
        prob2.run_model()

        fuelburndiff = np.abs(
            prob.get_val(point_name0 + ".fuelburn")[0] - prob2.get_val(point_name1 + ".fuelburn")[0]
        )

        assert_near_equal(fuelburndiff, 0.0, 1e-6)


if __name__ == "__main__":
    unittest.main()
