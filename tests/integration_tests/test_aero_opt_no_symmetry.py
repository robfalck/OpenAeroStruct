import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
import unittest


class Test(unittest.TestCase):
    def test(self):
        import numpy as np

        from openaerostruct.meshing.mesh_generator import generate_mesh
        from openaerostruct.geometry.geometry_group import Geometry
        from openaerostruct.aerodynamics.aero_groups import AeroPoint

        # Create a dictionary to store options about the mesh
        mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "CRM", "symmetry": False, "num_twist_cp": 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": False,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "fem_model_type": "tube",
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
            "with_viscous": True,  # if true, compute viscous drag
            "with_wave": False,  # if true, compute wave drag
        }

        # Create the OpenMDAO problem
        subsystems = {}
        connections = []

        # Create an independent variable component that will supply the flow
        # conditions to the problem.

        # Add this IndepVarComp to the problem model

        # Create and add a group that handles the geometry for the
        # aerodynamic lifting surface
        geom_group = Geometry(surface=surface)
        subsystems[surface["name"]] = geom_group

        # Create the aero point group, which contains the actual aerodynamic
        # analyses
        aero_group = AeroPoint(surfaces=[surface], promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])
        point_name = "aero_point_0"
        subsystems[point_name] = aero_group

        name = surface["name"]

        # Connect the mesh from the geometry component to the analysis point
        connections.append(om.Connection(src=name + ".mesh", tgt=point_name + "." + name + ".def_mesh"))

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        connections.append(om.Connection(src=name + ".mesh", tgt=point_name + ".aero_states." + name + "_def_mesh"))

        connections.append(om.Connection(src=name + ".t_over_c", tgt=point_name + "." + name + "_perf." + "t_over_c"))

        # Import the Scipy Optimizer and set the driver of the problem to use
        # it, which defaults to an SLSQP optimization method

        # Setup problem and add design variables, constraint, and objective

        # Set up and run the optimization problem
        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={"wing.twist_cp": DesignVar(lower=-10.0, upper=15.0)},
                constraints={point_name + ".wing_perf.CL": Constraint(equals=0.5)},
                objectives={
                    point_name + ".wing_perf.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))
                },
                optimizer=om.ScipyMinimizeOptimizer(tol=1e-9, optimizer="SLSQP"),
            )
        )
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 5.0, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")
        prob.run_model()

        prob.run_optimizer()

        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CD")[0], 0.03157639089274714, 1e-5)
        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CL")[0], 0.5, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -2.1184353479558586, 1e-4)


if __name__ == "__main__":
    unittest.main()
