import om4.api as om
from om4.utils.assert_utils import assert_near_equal
import unittest


class Test(unittest.TestCase):
    def test(self):
        # docs checkpoint 0
        import numpy as np

        from openaerostruct.meshing.mesh_generator import generate_mesh
        from openaerostruct.geometry.geometry_group import Geometry
        from openaerostruct.aerodynamics.aero_groups import AeroPoint

        # Create a dictionary to store options about the surface
        mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
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

        # Create a dictionary to store options about the surface
        mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "rect", "symmetry": True, "offset": np.array([50, 0.0, 0.0])}

        mesh = generate_mesh(mesh_dict)

        surf_dict2 = {
            # Wing definition
            "name": "tail",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "twist_cp": twist_cp,
            "mesh": mesh,
            # Aerodynamic performance of the lifting surface at
            # an angle of attack of 0 (alpha=0).
            # These CL0 and CD0 values are added to the CL and CD
            # obtained from aerodynamic analysis of the surface to get
            # the total CL and CD.
            # These CL0 and CD0 values do not vary wrt alpha.
            "CL0": 0.0,  # CL of the surface at alpha=0
            "CD0": 0.0,  # CD of the surface at alpha=0
            "fem_origin": 0.35,
            # Airfoil properties for viscous drag calculation
            "k_lam": 0.05,  # percentage of chord with laminar
            # flow, used for viscous drag
            "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
            "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
            # thickness
            "with_viscous": True,  # if true, compute viscous drag
            "with_wave": False,  # if true, compute wave drag
        }

        surfaces = [surf_dict, surf_dict2]

        # Create the problem and the model group
        subsystems = {}
        connections = []

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            geom_group = Geometry(surface=surface)

            # Add tmp_group to the problem as the name of the surface.
            # Note that is a group and performance group for each
            # individual surface.
            subsystems[surface["name"]] = geom_group

        # Loop through and add a certain number of aero points
        for i in range(1):
            # Create the aero point group and add it to the model
            aero_group = AeroPoint(surfaces=surfaces, promotes_inputs=["v", "alpha", "Mach_number", "re", "rho", "cg"])
            point_name = "aero_point_{}".format(i)
            subsystems[point_name] = aero_group

            # Connect flow properties to the analysis point

            # Connect the parameters within the model for each aero point
            for surface in surfaces:
                name = surface["name"]

                # Connect the mesh from the geometry component to the analysis point
                connections.append(om.Connection(src=name + ".mesh", tgt=point_name + "." + name + ".def_mesh"))

                # Perform the connections with the modified names within the
                # 'aero_states' group.
                connections.append(
                    om.Connection(src=name + ".mesh", tgt=point_name + ".aero_states." + name + "_def_mesh")
                )

                connections.append(
                    om.Connection(src=name + ".t_over_c", tgt=point_name + "." + name + "_perf." + "t_over_c")
                )

        # Set up the problem
        prob = om.Problem(model=om.Group(subsystems=subsystems, connections=connections))
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 5.0, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")

        prob.run_model()
        # docs checkpoint 1

        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CD")[0], 0.03539775778518085, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CL")[0], 0.5124736932248048, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -2.016825327289286, 1e-6)


if __name__ == "__main__":
    unittest.main()
