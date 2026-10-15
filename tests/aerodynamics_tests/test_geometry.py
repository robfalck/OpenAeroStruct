import unittest
import numpy as np

import om4.api as om

from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.utils.testing import run_test, get_default_surfaces, assert_near_equal


def _rich_surface(rng, S_ref_type):
    # This is a much richer test with the following attributes:
    # - 5x7 mesh so that each dimension of all variables is unique.
    # - Random values given as inputs.
    # - The mesh has been given a random z height at all points.
    mesh_dict = {"num_y": 7, "num_x": 5, "wing_type": "CRM", "symmetry": True, "num_twist_cp": 5}

    # Generate the aerodynamic mesh based on the previous dictionary
    mesh, twist_cp = generate_mesh(mesh_dict)

    mesh[:, :, 2] = rng.random(mesh[:, :, 2].shape)

    # Create a dictionary with info and options about the aerodynamic
    # lifting surface
    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": S_ref_type,  # how we compute the wing area,
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
    return surface, mesh


class Test(unittest.TestCase):
    def setUp(self):
        # setup a random generator and fix the seed
        self.rng = np.random.default_rng(1)

    def test(self):
        surfaces = get_default_surfaces()

        comp = VLMGeometry(surface=surfaces[0])

        run_test(self, comp, values={"comp.def_mesh": surfaces[0]["mesh"]})

    def test_derivs_wetted(self):
        # S_ref_type option is "wetted"
        surface, mesh = _rich_surface(self.rng, "wetted")

        comp = VLMGeometry(surface=surface)

        values = {"comp.def_mesh": mesh + self.rng.random(comp.inputs["def_mesh"].val.shape) * 0.1}

        run_test(self, comp, atol=5e-5, rtol=1e-5, values=values)

    def test_derivs_projected(self):
        # S_ref_type option is "projected"
        surface, mesh = _rich_surface(self.rng, "projected")

        comp = VLMGeometry(surface=surface)

        values = {"comp.def_mesh": mesh + self.rng.random(comp.inputs["def_mesh"].val.shape) * 0.1}

        run_test(self, comp, atol=5e-5, rtol=1e-5, values=values)

    def test_outputs(self):
        surfaces = get_default_surfaces()

        comp = VLMGeometry(surface=surfaces[0])

        prob = om.Problem(model=om.Group(subsystems={"geom": comp}))
        prob.set_val("geom.def_mesh", surfaces[0]["mesh"])
        prob.run_model()

        assert_near_equal(
            prob.get_val("geom.lengths_spanwise"), np.array([11.95624787, 11.90425878, 11.44086572]), 1e-6
        )
        assert_near_equal(prob.get_val("geom.widths"), np.array([9.7938336, 9.79384207, 9.79385053]), 1e-6)
        assert_near_equal(prob.get_val("geom.S_ref"), np.array([415.02211208]), 1e-6)
        assert_near_equal(prob.get_val("geom.chords"), np.array([2.72796, 5.1252628, 7.8891638, 13.6189974]), 1e-6)
        assert_near_equal(prob.get_val("geom.lengths"), np.array([2.72796, 5.1252628, 7.8891638, 13.6189974]), 1e-6)


if __name__ == "__main__":
    unittest.main()
