import unittest

import numpy as np

from openaerostruct.aerodynamics.mesh_point_forces import MeshPointForces
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()

        comp = MeshPointForces(surfaces=surfaces)

        run_test(self, comp)

    def test_derivatives(self):
        # For this test, we bump up the number of x and y points so that we are sure the
        # derivatives are correct.
        mesh_dict = {"num_y": 7, "num_x": 5, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)

        rng = np.random.default_rng(0)
        mesh[:, :, 2] = rng.random(mesh[:, :, 2].shape)

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
            # Wing definition
            "name": "wing",  # name of the surface
            "twist_cp": twist_cp,
            "mesh": mesh,
        }

        # surfaces = get_default_surfaces()
        surfaces = [surface]

        comp = MeshPointForces(surfaces=surfaces)

        values = {"comp.wing_sec_forces": rng.random(comp.inputs["wing_sec_forces"].val.shape)}

        run_test(self, comp, atol=3e-5, rtol=1e-5, values=values)


if __name__ == "__main__":
    unittest.main()
