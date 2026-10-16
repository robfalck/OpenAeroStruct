from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface
from openaerostruct.utils.vector_algebra import get_array_indices


class DisplacementTransfer(om.ExplicitComponent):
    """
    Apply the computed FEM displacements and rotations on the aerodynamic mesh
    to obtain the deformed mesh.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Original undeformed aerodynamic mesh.
    disp[ny, 6] : numpy array
        Displacements and rotations acting on the structural spar which come
        from solving the FEM system. Contains displacements for all six degrees
        of freedom, including displacements in the x, y, and z directions, and
        rotations about the x, y, and z axes.
    transformation_matrix[ny, 3, 3] : numpy array
        Array containing the transformation matrices to apply the rotations
        from the FEM results. These are the angles that come from the FEM
        solver that rotate the structural spar. They will rotate the mesh based
        on the both the angle (which comes from `disp`) and from the distance
        of the aerodynamic mesh nodes to the structural mesh nodes (basically
        a moment arm).
    nodes[ny, 3] : numpy array
        Coordinates of the FEM nodes.

    Returns
    -------
    def_mesh[nx, ny, 3] : numpy array
        The final deformed aerodynamic mesh for the lifting surface based on
        the FEM results.
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        nx, ny = f.surface["mesh"].shape[:2]
        d = VarDecl()

        d.add_input("mesh", val=np.ones((nx, ny, 3)), units="m")
        d.add_input("disp", val=np.ones((ny, 6)), units="m")
        d.add_input("transformation_matrix", val=np.ones((ny, 3, 3)))
        d.add_input("nodes", val=np.ones((ny, 3)), units="m")
        rng = np.random.default_rng(314)
        d.add_output("def_mesh", val=rng.random((nx, ny, 3)), units="m")

        # Create index arrays for each relevant input and output.
        # This allows us to set up the rows and cols for the sparse Jacobians.
        disp_indices = get_array_indices(ny, 6)
        nodes_indices = get_array_indices(ny, 3)
        mesh_indices = get_array_indices(nx, ny, 3)
        transform_indices = get_array_indices(ny, 3, 3)
        mesh_disp_indices = get_array_indices(nx, ny, 3)

        # Set up the rows and cols for `def_mesh` wrt `disp`
        rows = mesh_disp_indices.flatten()
        cols = np.einsum("i,jk->ijk", np.ones(nx), disp_indices[:, :3]).flatten()
        d.declare_partials("def_mesh", "disp", val=1.0, rows=rows, cols=cols)

        # Set up the rows and cols for `def_mesh` wrt `nodes`
        rows = np.einsum("ijk,l->ijkl", mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum("ik,jl->ijkl", np.ones((nx, 3), int), nodes_indices).flatten()
        d.declare_partials("def_mesh", "nodes", rows=rows, cols=cols)

        # Set up the rows and cols for `def_mesh` wrt `mesh`
        rows = np.einsum("ijk,l->ijkl", mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum("ijl,k->ijkl", mesh_indices, np.ones(3, int)).flatten()
        d.declare_partials("def_mesh", "mesh", rows=rows, cols=cols)

        # Set up the rows and cols for `def_mesh` wrt `transformation_matrix`
        rows = np.einsum("ijl,k->ijkl", mesh_disp_indices, np.ones(3, int)).flatten()
        cols = np.einsum("jlk,i->ijkl", transform_indices, np.ones(nx, int)).flatten()
        d.declare_partials("def_mesh", "transformation_matrix", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nodes = inputs["nodes"]
        mesh = inputs["mesh"]
        nx = mesh.shape[0]

        # Add the translational displacements, which are the x, y, z displacements applied to
        # all nodal mesh points.
        def_mesh = mesh + np.einsum("i,jk->ijk", np.ones(nx), inputs["disp"][:, :3])

        # Apply the transformation matrix to the moment arms from the structural mesh points to
        # the aerodynamic mesh points, to transfer the FEM rotations to the aerodynamic mesh.
        moment_arms = mesh - nodes
        def_mesh = def_mesh + np.einsum("lij,klj->kli", inputs["transformation_matrix"], moment_arms)

        outputs["def_mesh"] = def_mesh

    def compute_partials(self, inputs, partials):
        mesh = inputs["mesh"]
        nx, ny = mesh.shape[:2]
        transform = inputs["transformation_matrix"]

        partials["def_mesh", "nodes"] = -np.einsum("i,jlk->ijlk", np.ones(nx), transform).flatten()

        dmesh = np.einsum("i,jlk->ijlk", np.ones(nx), transform).flatten()
        dmesh += np.tile(np.eye(3), nx * ny).flatten(order="F")
        partials["def_mesh", "mesh"] = dmesh

        partials["def_mesh", "transformation_matrix"] = np.einsum(
            "ijk,l->ijkl", mesh - np.einsum("i,jk->ijk", np.ones(nx), inputs["nodes"]), np.ones(3)
        ).flatten()
