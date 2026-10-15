"""Unit tests for each geometry mesh transformation component."""

import unittest

import numpy as np

import om4.api as om

from openaerostruct.geometry.geometry_mesh_transformations import (
    Taper,
    ScaleX,
    Sweep,
    ShearX,
    Stretch,
    ShearY,
    Dihedral,
    ShearZ,
    Rotate,
)
from openaerostruct.meshing.mesh_generator import generate_mesh
from openaerostruct.utils.testing import assert_near_equal, run_test

# These have been chosen so that each dimension of the intermediate ndarrays is unique.
NY = 7
NX = 5


def get_mesh(symmetry):
    """
    Return a mesh for testing.
    """
    ny = (2 * NY - 1) if symmetry else NY

    # Create a dictionary to store options about the mesh
    mesh_dict = {
        "num_y": ny,
        "num_x": NX,
        "wing_type": "CRM",
        "symmetry": symmetry,
        "num_twist_cp": NY,
    }

    # Generate the aerodynamic mesh based on the previous dictionary
    mesh, twist_cp = generate_mesh(mesh_dict)

    surface = {}
    surface["symmetry"] = symmetry
    surface["type"] = "aero"

    # Random perturbations to the mesh so that we don't mask errors subtractively.
    rng = np.random.default_rng(0)
    mesh[:, :, 0] += 0.05 * rng.random(mesh[:, :, 2].shape)
    mesh[:, :, 1] += 0.05 * rng.random(mesh[:, :, 2].shape)
    mesh[:, :, 2] = rng.random(mesh[:, :, 2].shape)

    return mesh


class Test(unittest.TestCase):
    def setUp(self):
        # setup a random generator and fix the seed
        self.rng = np.random.default_rng(1)

    def test_taper(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())
        ref_axis_pos = float(self.rng.random())

        comp = Taper(val=val, mesh=mesh, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6)

    def test_taper_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())
        ref_axis_pos = float(self.rng.random())

        comp = Taper(val=val, mesh=mesh, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6)

    def test_taper_symmetry_one(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = 1.0
        ref_axis_pos = float(self.rng.random())

        comp = Taper(val=val, mesh=mesh, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6)

    def test_scalex(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)
        ref_axis_pos = float(self.rng.random())

        comp = ScaleX(val=val, mesh_shape=mesh.shape, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_scalex_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)
        ref_axis_pos = float(self.rng.random())

        comp = ScaleX(val=val, mesh_shape=mesh.shape, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_scalex_ref_axis_trailing_edge(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        # Test for chord_scaling_pos at trailing edge
        val = self.rng.random(NY)

        comp = ScaleX(val=val, mesh_shape=mesh.shape, ref_axis_pos=1.0)

        prob = om.Problem(model=om.Group(subsystems={"comp": comp}))
        prob.set_val("comp.in_mesh", mesh)
        prob.run_model()

        # If chord_scaling_pos = 1, TE should not move
        assert_near_equal(prob.get_val("comp.mesh")[-1, :, :], mesh[-1, :, :], tolerance=1e-10)

    def test_scalex_chord_value(self):
        # test actual chord value of a rectangular wing

        mesh_dict = {
            "num_y": 5,
            "num_x": 3,
            "wing_type": "rect",
            "symmetry": True,
            "span": 2.0,
            "root_chord": 0.2,
        }
        mesh_in = generate_mesh(mesh_dict)

        # initial chord
        chord_in = mesh_in[-1, 0, 0] - mesh_in[0, 0, 0]  # TE - LE

        comp = ScaleX(val=np.ones(3), mesh_shape=mesh_in.shape)

        prob = om.Problem(model=om.Group(subsystems={"comp": comp}))
        chord_scaling = 1.3
        prob.set_val("comp.chord", val=chord_scaling, units=None)  # apply chord scaling factor
        prob.set_val("comp.in_mesh", val=mesh_in, units="m")

        prob.run_model()

        # chord after manipulation
        mesh_out = prob.get_val("comp.mesh", units="m")
        chord_out = mesh_out[-1, 0, 0] - mesh_out[0, 0, 0]  # TE - LE

        assert_near_equal(chord_in * chord_scaling, chord_out, tolerance=1e-10)

    def test_sweep(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())

        comp = Sweep(val=val, mesh_shape=mesh.shape, symmetry=symmetry)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_sweep_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())

        comp = Sweep(val=val, mesh_shape=mesh.shape, symmetry=symmetry)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_shearx(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)

        comp = ShearX(val=val, mesh_shape=mesh.shape)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_stretch(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())
        ref_axis_pos = float(self.rng.random())

        comp = Stretch(val=val, mesh_shape=mesh.shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_stretch_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())
        ref_axis_pos = float(self.rng.random())

        comp = Stretch(val=val, mesh_shape=mesh.shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_sheary(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)

        comp = ShearY(val=val, mesh_shape=mesh.shape)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_dihedral(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = 15.0 * float(self.rng.random())

        comp = Dihedral(val=val, mesh_shape=mesh.shape, symmetry=symmetry)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_dihedral_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = float(self.rng.random())

        comp = Dihedral(val=val, mesh_shape=mesh.shape, symmetry=symmetry)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_shearz(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)

        comp = ShearZ(val=val, mesh_shape=mesh.shape)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_rotate(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)
        ref_axis_pos = float(self.rng.random())

        comp = Rotate(val=val, mesh_shape=mesh.shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_rotate_symmetry(self):
        symmetry = True
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)
        ref_axis_pos = float(self.rng.random())

        comp = Rotate(val=val, mesh_shape=mesh.shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos)

        run_test(self, comp, atol=1e-6, rtol=1e-6, values={"comp.in_mesh": mesh})

    def test_rotate_trailing_edge(self):
        symmetry = False
        mesh = get_mesh(symmetry)

        val = self.rng.random(NY)

        comp = Rotate(val=val, mesh_shape=mesh.shape, symmetry=symmetry, ref_axis_pos=1.0)

        prob = om.Problem(model=om.Group(subsystems={"comp": comp}))
        prob.set_val("comp.in_mesh", mesh)
        prob.run_model()

        # If chord_scaling_pos = 1, TE should not move
        assert_near_equal(prob.get_val("comp.mesh")[-1, :, :], mesh[-1, :, :], tolerance=1e-10)


if __name__ == "__main__":
    unittest.main()
