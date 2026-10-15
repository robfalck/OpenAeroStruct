import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.promotion import PromotesSpec
from om4.utils.assert_utils import assert_near_equal
import unittest


class Test(unittest.TestCase):
    def test(self):
        # docs checkpoint 0
        import numpy as np

        from openaerostruct.meshing.mesh_generator import generate_mesh
        from openaerostruct.geometry.geometry_group import Geometry
        from openaerostruct.aerodynamics.aero_groups import AeroPoint
        from openaerostruct.integration.multipoint_comps import MultiCD

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
        }

        surfaces = [surf_dict]

        n_points = 2

        subsystems = {}
        connections = []

        # Loop over each surface and create the geometry groups
        for surface in surfaces:
            # Get the surface name and create a group to contain components only for this surface.
            name = surface["name"]
            geom_group = Geometry(surface=surface)

            # Add geom_group to the problem with the name of the surface.
            subsystems[name + "_geom"] = geom_group

        # Loop through and add a certain number of aero points
        for i in range(n_points):
            # Create the aero point group and add it to the model. Each point takes one entry of the
            # promoted alpha vector.
            point_name = "aero_point_{}".format(i)
            subsystems[point_name] = AeroPoint(
                surfaces=surfaces,
                promotes_inputs=[
                    "v",
                    "Mach_number",
                    "re",
                    "rho",
                    "cg",
                    PromotesSpec(patterns=["alpha"], src_indices=[i], src_shape=(n_points,)),
                ],
            )

            # Connect the parameters within the model for each aero point
            for surface in surfaces:
                name = surface["name"]

                # Connect the drag coeff at each point to the multi_CD component, which does the summation.
                connections.append(om.Connection(src=point_name + ".CD", tgt="multi_CD." + str(i) + "_CD"))

                # Connect the mesh from the geometry component to the analysis point
                connections.append(om.Connection(src=name + "_geom.mesh", tgt=point_name + "." + name + ".def_mesh"))

                # Perform the connections with the modified names within the 'aero_states' group.
                connections.append(
                    om.Connection(src=name + "_geom.mesh", tgt=point_name + ".aero_states." + name + "_def_mesh")
                )
                connections.append(
                    om.Connection(src=name + "_geom.t_over_c", tgt=point_name + "." + name + "_perf." + "t_over_c")
                )

        subsystems["multi_CD"] = MultiCD(n_points=n_points, promotes_outputs=["CD"])

        # Design variables are the wing twist and angle-of-attack at each point. Each point has a
        # different target CL, and the objective is the sum of CDs at each point.
        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "alpha": DesignVar(lower=-15, upper=15),
                    "wing_geom.twist_cp": DesignVar(lower=-5, upper=8),
                },
                constraints={
                    "aero_point_0.wing_perf.CL": Constraint(equals=0.45),
                    "aero_point_1.wing_perf.CL": Constraint(equals=0.5),
                },
                objectives={"CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))},
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP", tol=1e-9),
            )
        )
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", np.ones(n_points) * 6.64, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")

        prob.run_optimizer()
        # docs checkpoint 1

        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CL")[0], 0.45, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CD")[0], 0.030502881726099183, 1e-6)
        assert_near_equal(prob.get_val("aero_point_1.wing_perf.CL")[0], 0.5, 1e-6)
        assert_near_equal(prob.get_val("aero_point_1.wing_perf.CD")[0], 0.03195303546890328, 1e-6)


if __name__ == "__main__":
    unittest.main()
