"""Manipulate geometry mesh based on high-level design parameters."""

from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.structures.utils import radii
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class RadiusComp(om.ExplicitComponent):
    """
    Compute the radius of a structural spar based on the mesh and thickness over
    chord ratio.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface..
    t_over_c[ny-1] : numpy array
        The streamwise thickness-to-chord ratio of each VLM panel.

    Returns
    -------
    radius[ny-1] : numpy array
        Radius of each element of the FEM spar.

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

        d.add_input("mesh", val=np.zeros((nx, ny, 3)), units="m")
        d.add_input("t_over_c", val=np.ones((ny - 1)))
        d.add_output("radius", val=np.ones((ny - 1)), units="m")

        arange = np.arange(ny - 1)
        d.declare_partials("radius", "t_over_c", rows=arange, cols=arange)

        row = np.tile(np.zeros(6), ny - 1) + np.repeat(arange, 6)
        rows = np.concatenate([row, row])
        col = np.tile(np.arange(6), ny - 1) + np.repeat(3 * arange, 6)
        cols = np.concatenate([col, col + (nx - 1) * 3 * ny])

        d.declare_partials("radius", "mesh", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["radius"] = radii(inputs["mesh"], inputs["t_over_c"])

    def compute_partials(self, inputs, partials):
        """
        Obtain the radii of the FEM element based on local chord.
        """
        mesh = inputs["mesh"]
        ny = mesh.shape[1]
        vectors = mesh[-1, :, :] - mesh[0, :, :]
        chords = np.sqrt(np.sum(vectors**2, axis=1))
        t_c = inputs["t_over_c"]

        dr_dtoc = 0.25 * (chords[:-1] + chords[1:])
        partials["radius", "t_over_c"] = dr_dtoc

        dr_dchords = 0.25 * t_c
        dr = mesh[0, :] - mesh[-1, :]

        length = np.sqrt(np.sum(dr**2, axis=1))
        dr = dr / length[:, np.newaxis]

        drad = np.empty((ny - 1, 6))
        drad[:, :3] = np.einsum("i, ij -> ij", dr_dchords, dr[:-1, :])
        drad[:, 3:] = np.einsum("i, ij -> ij", dr_dchords, dr[1:, :])
        drad = drad.flatten()

        partials["radius", "mesh"] = np.concatenate([drad, -drad])
