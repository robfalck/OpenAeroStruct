from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface
from openaerostruct.utils.vector_algebra import get_array_indices


class ComputeTransformationMatrix(om.ExplicitComponent):
    """
    Compute the transformation matrix used to apply the rotations obtained
    from the FEM system to the aerodynamic mesh.

    Technically the order of rotations (x, y, z) matter, but here we assume
    they are independent. This allows us to construct a single transformation
    matrix and apply it to the moment arms going from the structural mesh
    to the aerodynamic mesh.

    Parameters
    ----------
    disp[ny, 6] : numpy array
        Displacements and rotations acting on the structural spar which come
        from solving the FEM system. Contains displacements for all six degrees
        of freedom, including displacements in the x, y, and z directions, and
        rotations about the x, y, and z axes.

    Returns
    -------
    transformation_matrix[ny, 3, 3] : numpy array
        Array containing the transformation matrices to apply the rotations
        from the FEM results. These are the angles that come from the FEM
        solver that rotate the structural spar. They will rotate the mesh based
        on the both the angle (which comes from `disp`) and from the distance
        of the aerodynamic mesh nodes to the structural mesh nodes (basically
        a moment arm).
    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("disp", val=np.zeros((ny, 6)), units="m")
        d.add_output("transformation_matrix", val=np.zeros((ny, 3, 3)))

        # Create index arrays for each relevant input and output.
        # This allows us to set up the rows and cols for the sparse Jacobians.
        transform_indices = get_array_indices(ny, 3, 3)

        # Set up the rows and cols for `transformation_matrix` wrt `disp`
        rows = np.einsum("ijk,l->ijkl", transform_indices, np.ones(3, int)).flatten()
        cols = np.einsum("il,jk->ijkl", get_array_indices(ny, 6)[:, 3:], np.ones((3, 3), int)).flatten()
        d.declare_partials("transformation_matrix", "disp", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        disp = inputs["disp"]
        ny = disp.shape[0]

        # These are the rotations obtained from the FEM solution
        rx = disp[:, 3]
        ry = disp[:, 4]
        rz = disp[:, 5]

        # The diagonal of each transformation matrix accounts for the equivalent identity matrices
        # that are added in later steps, because the three rotational matrices are treated as
        # independent and combined by adding.
        T = np.zeros((ny, 3, 3), dtype=disp.dtype)
        for i in range(3):
            T[:, i, i] -= 2.0

        # T[ 1:,  1:] += [[cos(rx), -sin(rx)], [ sin(rx), cos(rx)]]
        T[:, 1, 1] += np.cos(rx)
        T[:, 1, 2] -= np.sin(rx)
        T[:, 2, 1] += np.sin(rx)
        T[:, 2, 2] += np.cos(rx)

        # T[::2, ::2] += [[cos(ry),  sin(ry)], [-sin(ry), cos(ry)]]
        T[:, 0, 0] += np.cos(ry)
        T[:, 0, 2] += np.sin(ry)
        T[:, 2, 0] -= np.sin(ry)
        T[:, 2, 2] += np.cos(ry)

        # T[ :2,  :2] += [[cos(rz), -sin(rz)], [ sin(rz), cos(rz)]]
        T[:, 0, 0] += np.cos(rz)
        T[:, 0, 1] -= np.sin(rz)
        T[:, 1, 0] += np.sin(rz)
        T[:, 1, 1] += np.cos(rz)

        outputs["transformation_matrix"] = T

    def compute_partials(self, inputs, partials):
        disp = inputs["disp"]
        ny = disp.shape[0]
        rx = disp[:, 3]
        ry = disp[:, 4]
        rz = disp[:, 5]

        # Each entry of the matrix is a sum of sines and cosines, so the derivatives are the
        # trigonometric derivatives of those terms.
        derivs = np.zeros((ny, 3, 3, 3))

        derivs[:, 1, 1, 0] -= np.sin(rx)
        derivs[:, 1, 2, 0] -= np.cos(rx)
        derivs[:, 2, 1, 0] += np.cos(rx)
        derivs[:, 2, 2, 0] -= np.sin(rx)

        derivs[:, 0, 0, 1] -= np.sin(ry)
        derivs[:, 0, 2, 1] += np.cos(ry)
        derivs[:, 2, 0, 1] -= np.cos(ry)
        derivs[:, 2, 2, 1] -= np.sin(ry)

        derivs[:, 0, 0, 2] -= np.sin(rz)
        derivs[:, 0, 1, 2] -= np.cos(rz)
        derivs[:, 1, 0, 2] += np.cos(rz)
        derivs[:, 1, 1, 2] -= np.sin(rz)

        partials["transformation_matrix", "disp"] = derivs.flatten()
