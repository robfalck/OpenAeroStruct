import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
import numpy as np
import unittest

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.multipoint_comps import MultiCD


class Test(unittest.TestCase):
    def test(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {
            "num_y": 5,
            "num_x": 3,
            "wing_type": "CRM",
            "symmetry": True,
            "num_twist_cp": 5,
            "span_cos_spacing": 0.0,
        }

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "fem_model_type": "tube",
            "mesh": mesh,
            "twist_cp": twist_cp,
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
            "with_viscous": True,  # if true, compute viscous drag
            "with_wave": False,  # if true, compute wave drag
            "span": 10.0,
        }

        surfaces = [surf_dict]

        n_points = 2

        subsystems = {}
        connections = []

        # Loop through and add a certain number of aero points
        for i in range(n_points):
            # Each flight point gets its own geometry group so the wing can morph between points. The
            # twist control points are separate design variables per point, and the other
            # geometric inputs are shared.
            point_name = "aero_point_{}".format(i)
            subsystems[point_name] = AeroPoint(
                surfaces=surfaces, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"]
            )

            # Connect the parameters within the model for each aero point
            for surface in surfaces:
                name = surface["name"]
                geom_name = point_name + "_" + name + "_geom"
                subsystems[geom_name] = Geometry(
                    surface=surface,
                    connect_geom_DVs=False,
                    promotes_inputs=[("twist_cp", "twist_cp_{}".format(i)), "t_over_c_cp", "span"],
                )

                connections.append(om.Connection(src=point_name + ".CD", tgt="multi_CD." + str(i) + "_CD"))

                # Connect the mesh from the geometry component to the analysis point
                connections.append(om.Connection(src=geom_name + ".mesh", tgt=point_name + "." + name + ".def_mesh"))

                # Perform the connections with the modified names within the
                # 'aero_states' group.
                connections.append(
                    om.Connection(src=geom_name + ".mesh", tgt=point_name + ".aero_states." + name + "_def_mesh")
                )

                connections.append(
                    om.Connection(src=geom_name + ".t_over_c", tgt=point_name + "." + name + "_perf." + "t_over_c")
                )

        subsystems["multi_CD"] = MultiCD(n_points=n_points, promotes_outputs=["CD"])

        # Set up the problem
        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "alpha": DesignVar(lower=-15, upper=15),
                    "twist_cp_0": DesignVar(lower=-5, upper=8),
                    "twist_cp_1": DesignVar(lower=-5, upper=8),
                    "span": DesignVar(lower=2, upper=15),
                },
                constraints={
                    "aero_point_0.wing_perf.CL": Constraint(equals=0.45),
                    "aero_point_1.wing_perf.CL": Constraint(equals=0.50),
                },
                objectives={"CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-9),
            )
        )
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 6.64, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")
        prob.set_val("t_over_c_cp", np.array([0.15]))
        prob.set_val("span", 12.0, units="m")
        prob.set_val("twist_cp_0", np.zeros((5)), units="deg")
        prob.set_val("twist_cp_1", np.zeros((5)), units="deg")

        prob.run_optimizer()

        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CL")[0], 0.45, 1e-6)
        assert_near_equal(prob.get_val("aero_point_1.wing_perf.CL")[0], 0.5, 1e-6)
        assert_near_equal(
            prob.get_val("twist_cp_0"), np.array([8.0, -1.28899415, -2.57798831, -1.28899415, -0.96161429]), 1e-6
        )
        assert_near_equal(
            prob.get_val("twist_cp_1"), np.array([8.0, -0.02636487, -0.05272973, -0.02636487, 0.7889681]), 1e-6
        )
        assert_near_equal(prob.get_val("aero_point_1.wing_perf.CL")[0], 0.5, 1e-6)


if __name__ == "__main__":
    unittest.main()
