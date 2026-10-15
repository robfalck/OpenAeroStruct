import om4.api as om
from om4.variables.design_var import DesignVar
from om4.variables.response import Constraint, Objective
from om4.transforms.transform import AffineTransform
from om4.utils.assert_utils import assert_near_equal
import unittest


class Test(unittest.TestCase):
    def test(self):
        # docs checkpoint 0
        import numpy as np

        from openaerostruct.meshing.mesh_generator import generate_mesh
        from openaerostruct.geometry.geometry_group import Geometry
        from openaerostruct.aerodynamics.aero_groups import AeroPoint

        # Instantiate the problem and the model group
        subsystems = {}
        connections = []

        # docs checkpoint 1

        # docs checkpoint 2
        # Create a dictionary to store options about the surface
        mesh_dict = {
            "num_y": 5,
            "num_x": 3,
            "wing_type": "rect",
            "symmetry": True,
            "span": 10.0,
            "chord": 1,
            "span_cos_spacing": 1.0,
        }

        mesh = generate_mesh(mesh_dict)

        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "twist_cp": np.zeros(2),
            "mesh": mesh,
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0,  # CD of the surface at alpha=0
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c_cp": np.array([0.12]),  # thickness over chord ratio (NACA0015)
            "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
            # thickness
            "with_viscous": False,  # if true, compute viscous drag,
            "with_wave": False,  # if true, compute wave drag
            "sweep": 0.0,
            "dihedral": 0.0,
            "taper": 1.0,
        }  # end of surface dictionary
        # docs checkpoint 3

        # docks checkpoint 4
        geom_group = Geometry(surface=surface)

        # Add tmp_group to the problem as the name of the surface.
        # Note that is a group and performance group for each
        # individual surface.
        subsystems[surface["name"]] = geom_group

        # Create the aero point group and add it to the model
        aero_group = AeroPoint(surfaces=[surface], promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])
        point_name = "aero_point_0"
        subsystems[point_name] = aero_group

        # Connect flow properties to the analysis point

        name = "wing"

        # Connect the mesh from the geometry component to the analysis point
        connections.append(om.Connection(src=name + ".mesh", tgt=point_name + "." + name + ".def_mesh"))

        # Perform the connections with the modified names within the
        # 'aero_states' group.
        connections.append(om.Connection(src=name + ".mesh", tgt=point_name + ".aero_states." + name + "_def_mesh"))

        connections.append(om.Connection(src=name + ".t_over_c", tgt=point_name + "." + name + "_perf." + "t_over_c"))

        # # Setup problem and add design variables, constraint, and objective

        # Set up the problem
        prob = om.Problem(
            model=om.Group(
                subsystems=subsystems,
                connections=connections,
                design_vars={
                    "wing.twist_cp": DesignVar(lower=-10.0, upper=15.0),
                    "wing.sweep": DesignVar(lower=-10.0, upper=30.0),
                    "wing.dihedral": DesignVar(lower=-10.0, upper=15.0),
                },
                constraints={point_name + ".wing_perf.CL": Constraint(equals=0.5)},
                objectives={
                    point_name + ".wing_perf.CD": Objective(transform=AffineTransform.from_scaler_adder(scaler=1e4))
                },
                optimizer=om.ScipyMinimizeOptimizer(optimizer="SLSQP"),
            )
        )
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 5.0, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")

        prob.run_optimizer()
        # docs checkpoint 5

        assert_near_equal(prob.get_val("aero_point_0.CD")[0], 0.004938282205728244, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CL")[0], 0.5, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.7630284313966209, 1e-6)
        assert_near_equal(prob.get_val("wing.dihedral")[0], -2.9553117192703464, 1e-6)
        assert_near_equal(prob.get_val("wing.sweep")[0], 30.0, 1e-6)
        np.testing.assert_allclose(
            prob.get_val("wing.twist_cp"), [-2.8855929739789588, 4.483359585932103], rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
