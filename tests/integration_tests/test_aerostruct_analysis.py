import om4.api as om
from om4.utils.assert_utils import assert_near_equal
import unittest
import numpy as np

from openaerostruct.meshing.mesh_generator import generate_mesh

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from openaerostruct.utils.constants import grav_constant


class Test(unittest.TestCase):
    def test(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {"num_y": 5, "num_x": 2, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

        mesh, twist_cp = generate_mesh(mesh_dict)

        surf_dict = {
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

        surfaces = [surf_dict]

        # Create the problem and assign the model group
        subsystems = {}
        connections = []

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            # Get the surface name and create a group to contain components
            # only for this surface
            name = surface["name"]

            aerostruct_group = AerostructGeometry(surface=surface)

            # Add tmp_group to the problem with the name of the surface.
            subsystems[name] = aerostruct_group

        # Loop through and add a certain number of aero points
        for i in range(1):
            point_name = "AS_point_{}".format(i)
            # Connect the parameters within the model for each aero point

            # Create the aero point group and add it to the model
            AS_point = AerostructPoint(
                surfaces=surfaces,
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

            subsystems[point_name] = AS_point

            # Connect flow properties to the analysis point

            for _surface in surfaces:
                com_name = point_name + "." + name + "_perf"
                connections.append(
                    om.Connection(
                        src=name + ".local_stiff_transformed",
                        tgt=point_name + ".coupled." + name + ".local_stiff_transformed",
                    )
                )
                connections.append(om.Connection(src=name + ".nodes", tgt=point_name + ".coupled." + name + ".nodes"))

                # Connect aerodyamic mesh to coupled group mesh
                connections.append(om.Connection(src=name + ".mesh", tgt=point_name + ".coupled." + name + ".mesh"))

                # Connect performance calculation variables
                connections.append(om.Connection(src=name + ".radius", tgt=com_name + ".radius"))
                connections.append(om.Connection(src=name + ".thickness", tgt=com_name + ".thickness"))
                connections.append(om.Connection(src=name + ".nodes", tgt=com_name + ".nodes"))
                connections.append(
                    om.Connection(
                        src=name + ".cg_location", tgt=point_name + "." + "total_perf." + name + "_cg_location"
                    )
                )
                connections.append(
                    om.Connection(
                        src=name + ".structural_mass", tgt=point_name + "." + "total_perf." + name + "_structural_mass"
                    )
                )
                connections.append(om.Connection(src=name + ".t_over_c", tgt=com_name + ".t_over_c"))

        # Set up the problem
        prob = om.Problem(model=om.Group(subsystems=subsystems, connections=connections))
        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 5.0, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("CT", grav_constant * 17.0e-6, units="1/s")
        prob.set_val("R", 11.165e6, units="m")
        prob.set_val("W0", 0.4 * 3e5, units="kg")
        prob.set_val("speed_of_sound", 295.4, units="m/s")
        prob.set_val("load_factor", 1.0)
        prob.set_val("empty_cg", np.zeros((3)), units="m")

        prob.run_model()

        assert_near_equal(prob.get_val("AS_point_0.fuelburn")[0], 241347.34494621187, 1e-4)
        assert_near_equal(prob.get_val("AS_point_0.CM")[1], -0.7040800097095121, 1e-5)


if __name__ == "__main__":
    unittest.main()
