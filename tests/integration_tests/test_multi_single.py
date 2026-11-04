import unittest

import numpy as np

import om4.api as om
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
from om4.utils.promotion import PromotesSpec
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import Geometry, MultiSecGeometry
from openaerostruct.geometry.multi_section import build_section_dicts, scalar_overlay, unify_mesh
from openaerostruct.utils.testing import get_single_section_surface, get_two_section_surface


class Test(unittest.TestCase):
    def test(self):
        """Create a dictionary with info and options about the aerodynamic
        single section lifting surface"""
        surfaceSingle = get_single_section_surface()

        """Create a dictionary with info and options about the multi-section aerodynamic
        lifting surface"""
        surfaceMulti, sec_chord_cp = get_two_section_surface()
        surfaceMulti["name"] = "surfaceMulti"

        # Generate the sections and unified mesh here in addition to adding the components.
        # This has to also be done here since AeroPoint has to know the unified mesh size.
        section_surfaces = build_section_dicts(surfaceMulti)
        uni_mesh = unify_mesh(section_surfaces)
        surfaceMulti["mesh"] = uni_mesh

        subsystems = {
            "surface": Geometry(surface=surfaceSingle),
            "aero_point_0": AeroPoint(
                surfaces=[surfaceSingle],
                promotes_inputs=[
                    "v",
                    "Mach_number",
                    "re",
                    "rho",
                    "cg",
                    PromotesSpec(patterns=["alpha"], src_indices=[0], src_shape=(2,)),
                ],
            ),
            "surfaceMulti": MultiSecGeometry(
                surface=surfaceMulti, joining_comp=True, dim_constr=[np.array([1, 0, 0]), np.array([1, 0, 0])]
            ),
            "aero_point_1": AeroPoint(
                surfaces=[scalar_overlay(surfaceMulti)],
                promotes_inputs=[
                    "v",
                    "Mach_number",
                    "re",
                    "rho",
                    "cg",
                    PromotesSpec(patterns=["alpha"], src_indices=[1], src_shape=(2,)),
                ],
            ),
        }

        connections = [
            # Connect the mesh from the single-section geometry component to its analysis point.
            om.Connection(src="surface.mesh", tgt="aero_point_0.surface.def_mesh"),
            om.Connection(src="surface.mesh", tgt="aero_point_0.aero_states.surface_def_mesh"),
            # Connect the unified mesh from the mesh unification component to its analysis point.
            om.Connection(
                src="surfaceMulti.surfaceMulti_unification.surfaceMulti_uni_mesh",
                tgt="aero_point_1.surfaceMulti.def_mesh",
            ),
            om.Connection(
                src="surfaceMulti.surfaceMulti_unification.surfaceMulti_uni_mesh",
                tgt="aero_point_1.aero_states.surfaceMulti_def_mesh",
            ),
        ]

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={"alpha": DesignVar(lower=0.0, upper=10.0)},
                constraints={
                    "aero_point_0.CL": Constraint(equals=0.3),
                    "aero_point_1.CL": Constraint(equals=0.3),
                },
                objectives={"aero_point_0.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-6, maxiter=1000),
            )
        )

        prob.set_val("v", 1.0, units="m/s")
        prob.set_val("alpha", 10.0 * np.ones(2), units="deg")
        prob.set_val("Mach_number", 0.3)
        prob.set_val("re", 1.0e5, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros(3), units="m")

        prob.run_optimizer()
        # Workaround for om4 B-018 (ai/OM4_NEEDS.md): CD isn't part of any declared
        # design_var/constraint/objective here (only CL is), so it can be left stale by
        # run_optimizer(); force a full re-propagation before reading it.
        prob.run_model()

        CDDiff = np.abs(
            prob.get_val("aero_point_1.surfaceMulti_perf.CD")[0] - prob.get_val("aero_point_0.surface_perf.CD")[0]
        )
        CLDiff = np.abs(
            prob.get_val("aero_point_1.surfaceMulti_perf.CL")[0] - prob.get_val("aero_point_0.surface_perf.CL")[0]
        )

        assert_near_equal(CDDiff, 0.0, 1e-6)
        assert_near_equal(CLDiff, 0.0, 1e-6)


if __name__ == "__main__":
    unittest.main()
