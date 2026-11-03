import numpy as np
import unittest

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant

import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform


"""
This is not a physically meaningful case, in that it is not set up to give
the correct results or with the correct coupled physics between the two points.
Instead, this is simply testing the morphing aspect of the code using a
multipoint aerostruct formulation.

Each flight point gets its own top-level AerostructGeometry/AerostructPoint pair
(connect_geom_DVs=False) so the wing can morph independently between points; the
twist and thickness control points are per-point design variables, set directly on
each point's unpromoted path rather than through a shared IndepVarComp.
"""


class Test(unittest.TestCase):
    def test(self):
        mesh_dict = {"num_y": 11, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "fem_model_type": "tube",
            "thickness_cp": np.array([0.1, 0.2, 0.3]),
            "twist_cp": twist_cp,
            "mesh": mesh,
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.015,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
            "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
            # thickness
            "with_viscous": True,
            "with_wave": False,  # if true, compute wave drag
            # Structural values are based on aluminum 7075
            "E": 70.0e9,  # [Pa] Young's modulus of the spar
            "G": 30.0e9,  # [Pa] shear modulus of the spar
            "yield": 500.0e6,
            "safety_factor": 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
            "mrho": 3.0e3,  # [kg/m^3] material density
            "fem_origin": 0.35,  # normalized chordwise location of the spar
            "wing_weight_ratio": 2.0,
            "struct_weight_relief": False,  # True to add the weight of the structure to the loads on the structure
            "distributed_fuel_weight": False,
            # Constraints
            "exact_failure_constraint": False,  # if false, use KS function
        }

        name = "wing"
        n_points = 2

        # Per-point flow conditions (these differ between points, unlike OM3's shared,
        # sliced IndepVarComp vectors).
        v = [248.136, 0.5 * 340.0]
        alpha = [5.0, 10.0]
        mach_number = [0.84, 0.5]
        re = [1.0e6, 0.5e6]
        rho = [0.38, 0.764]
        load_factor = [1.0, 2.5]
        twist_cp_pts = [
            np.array([2.0, 3.0, 4.0, 4.0, 4.0]),
            np.array([4.0, 4.0, 4.0, 5.0, 6.0]),
        ]

        subsystems = {}
        connections = []
        design_vars = {}
        constraints = {}

        for i in range(n_points):
            geom_name = "wing_point{}".format(i)
            point_name = "AS_point_{}".format(i)

            subsystems[geom_name] = AerostructGeometry(surface=surface, connect_geom_DVs=False)

            subsystems[point_name] = AerostructPoint(
                surfaces=[surface],
                promotes_inputs=["CT", "R", "W0", "speed_of_sound", "empty_cg"],
            )

            com_name = point_name + "." + name + "_perf"
            connections += [
                om.Connection(
                    src=geom_name + ".local_stiff_transformed",
                    tgt=point_name + ".coupled." + name + ".local_stiff_transformed",
                ),
                om.Connection(src=geom_name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"),
                # Connect aerodyamic mesh to coupled group mesh
                om.Connection(src=geom_name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"),
                # Connect performance calculation variables
                om.Connection(src=geom_name + ".radius", tgt=com_name + ".radius"),
                om.Connection(src=geom_name + ".thickness", tgt=com_name + ".thickness"),
                om.Connection(src=geom_name + ".nodes", tgt=com_name + ".nodes"),
                om.Connection(src=geom_name + ".cg_location", tgt=point_name + ".total_perf." + name + "_cg_location"),
                om.Connection(
                    src=geom_name + ".structural_mass", tgt=point_name + ".total_perf." + name + "_structural_mass"
                ),
                # With connect_geom_DVs=False, the geometry subsystem's t_over_c output is not
                # auto-wired to tube_group's t_over_c input via shared promotion, so wire both
                # consumers explicitly. tube_group's t_over_c (fed by radius_comp) is itself
                # promoted all the way up to the geometry group's own boundary.
                om.Connection(src=geom_name + ".geometry.t_over_c", tgt=geom_name + ".t_over_c"),
                om.Connection(src=geom_name + ".geometry.t_over_c", tgt=com_name + ".t_over_c"),
            ]

            design_vars[geom_name + ".geometry.twist_cp"] = DesignVar(lower=-10.0, upper=15.0)
            design_vars[geom_name + ".tube_group.thickness_cp"] = DesignVar(
                lower=0.01, upper=0.5, transform=AffineTransform.from_scaler_adder(scaler=1e2)
            )
            design_vars[point_name + ".alpha"] = DesignVar(lower=-15.0, upper=15.0)

            constraints[point_name + "." + name + "_perf.failure"] = Constraint(upper=0.0)
            constraints[point_name + "." + name + "_perf.thickness_intersects"] = Constraint(upper=0.0)
            constraints[point_name + ".L_equals_W"] = Constraint(equals=0.0)

        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars=design_vars,
                constraints=constraints,
                objectives={
                    "AS_point_0.fuelburn": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e-5))
                },
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-9, maxiter=2),
            )
        )

        # Shared scalar flow/mass properties, common to both points.
        prob.set_val("CT", grav_constant * 17.0e-6, units="1/s")
        prob.set_val("R", 11.165e6, units="m")
        prob.set_val("W0", 0.4 * 3e5, units="kg")
        prob.set_val("speed_of_sound", 295.4, units="m/s")
        prob.set_val("empty_cg", np.zeros((3)), units="m")

        for i in range(n_points):
            geom_name = "wing_point{}".format(i)
            point_name = "AS_point_{}".format(i)

            prob.set_val(point_name + ".v", v[i], units="m/s")
            prob.set_val(point_name + ".alpha", alpha[i], units="deg")
            prob.set_val(point_name + ".Mach_number", mach_number[i])
            prob.set_val(point_name + ".re", re[i], units="1/m")
            prob.set_val(point_name + ".rho", rho[i], units="kg/m**3")
            prob.set_val(point_name + ".load_factor", load_factor[i])

            prob.set_val(geom_name + ".geometry.t_over_c_cp", np.array([0.15]))
            prob.set_val(geom_name + ".tube_group.thickness_cp", np.array([0.01, 0.01, 0.01]), units="m")
            prob.set_val(geom_name + ".geometry.twist_cp", twist_cp_pts[i], units="deg")

        prob.run_model()

        # Check the partials at the initial point in the design space,
        # only care about relative error
        data = prob.check_partials(differentiator="cs", only_incorrect=True, atol=1e20, rtol=1e-6)
        assert not data, {c: {k: (v["abs_error"], v["rel_error"]) for k, v in d.items()} for c, d in data.items()}

        # Run the optimizer for 2 iterations
        prob.run_optimizer()

        # Check the partials at this point in the design space
        data = prob.check_partials(differentiator="cs", only_incorrect=True, atol=1e20, rtol=1e-6)
        assert not data, {c: {k: (v["abs_error"], v["rel_error"]) for k, v in d.items()} for c, d in data.items()}


if __name__ == "__main__":
    unittest.main()
