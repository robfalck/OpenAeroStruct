import unittest

import numpy as np

import om4.api as om
from om4.utils.assert_utils import assert_near_equal

from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint


FLOW_VARS = ["v", "alpha", "Mach_number", "re", "rho", "cg"]


class Test(unittest.TestCase):
    def test(self):
        # Create a dictionary to store options about the surface
        mesh_dict = {"num_y": 5, "num_x": 2, "wing_type": "rect", "symmetry": True}

        mesh = generate_mesh(mesh_dict)

        surf_dict = {
            # Wing definition
            "name": "wing",  # name of the surface
            "symmetry": True,  # if true, model one half of wing
            # reflected across the plane y = 0
            "S_ref_type": "wetted",  # how we compute the wing area,
            # can be 'wetted' or 'projected'
            "twist_cp": np.array([0.0]),
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
        }

        surfaces = [surf_dict]

        subsystems = {}
        connections = []

        # Loop over each surface in the surfaces list
        for surface in surfaces:
            # Add a geometry group named for the surface.
            subsystems[surface["name"]] = Geometry(surface=surface)

        # Loop through and add a certain number of aero points
        for i in range(1):
            point_name = "aero_point_{}".format(i)

            # The flow conditions are promoted to the model, where they are set below.
            # (OM3 used an IndepVarComp; om4 sets unconnected inputs directly.)
            subsystems[point_name] = om.Subsystem(AeroPoint(surfaces=surfaces), promotes_inputs=FLOW_VARS)

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

        prob = om.Problem(model=om.Group(subsystems=subsystems, connections=connections))

        prob.set_val("v", 248.136, units="m/s")
        prob.set_val("alpha", 5.0, units="deg")
        prob.set_val("Mach_number", 0.84)
        prob.set_val("re", 1.0e6, units="1/m")
        prob.set_val("rho", 0.38, units="kg/m**3")
        prob.set_val("cg", np.zeros((3)), units="m")

        prob.run_model()

        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CD")[0], 0.03487336411850356, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.wing_perf.CL")[0], 0.4615561217697067, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[0], 0.0, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[1], -0.11507021674483686, 1e-6)
        assert_near_equal(prob.get_val("aero_point_0.CM")[2], 0.0, 1e-6)


if __name__ == "__main__":
    unittest.main()
