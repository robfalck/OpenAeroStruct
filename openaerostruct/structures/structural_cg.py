from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class StructuralCG(om.ExplicitComponent):
    """Compute center-of-gravity location of the spar elements.

    Parameters
    ----------
    A[ny-1] : numpy array
        Areas for each FEM element.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    structural_mass : float
        Weight of the structural spar.
    cg_location[3] : numpy array
        Location of the structural spar's cg.

    """

    surface: Surface
    check_partials_options: om.CheckPartialsOptions | None = om.CheckPartialsOptions(differentiator="cs")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        # Setup Inputs
        d.add_input("nodes", val=np.zeros((ny, 3)), units="m")
        d.add_input("structural_mass", val=0.0, units="kg")
        d.add_input("element_mass", val=np.zeros((ny - 1)), units="kg")

        # Setup Outputs
        d.add_output("cg_location", val=np.zeros((3)), units="m")

        # Setup Partials
        d.declare_partials(of="cg_location", wrt="structural_mass")
        d.declare_partials(of="cg_location", wrt="element_mass")
        # Setup Sparce Matrix
        dimensions = 3
        cols_const = np.arange(0, ny * dimensions, dimensions)
        rows = np.empty(ny * dimensions)
        cols = np.empty(ny * dimensions)
        for i in range(dimensions):
            rows[i * ny : i * ny + ny] = i
            cols[i * ny : i * ny + ny] = cols_const + i

        d.declare_partials(of="cg_location", wrt="nodes", rows=rows, cols=cols)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nodes = inputs["nodes"]
        structural_mass = inputs["structural_mass"]
        element_mass = inputs["element_mass"]

        # Calculate the center-of-gravity location of the spar elements only
        center_of_elements = ((nodes[1:, :] + nodes[:-1, :]) / 2.0).T
        cg_loc = center_of_elements.dot(element_mass) / structural_mass

        # If the tube is symmetric, double the computed weight and set the
        # y-location of the cg to 0, at the symmetry plane
        if self.surface["symmetry"]:
            cg_loc[1] = 0.0
            cg_loc *= 2.0

        outputs["cg_location"] = cg_loc

    def compute_partials(self, inputs, J):
        nodes = inputs["nodes"]
        ny = nodes.shape[0]
        structural_mass = inputs["structural_mass"]
        element_mass = em = inputs["element_mass"]

        center_of_elements = ((nodes[1:, :] + nodes[:-1, :]) / 2.0).T

        sum_coe_dot_em = center_of_elements.dot(element_mass)

        is_sym = self.surface["symmetry"]

        dcg_dmass = -sum_coe_dot_em / structural_mass**2
        dcg_dem = center_of_elements / structural_mass

        # derivates with respect to nodes require special ninja math
        values = np.zeros(ny)
        values[0 : ny - 1] = em
        values[1:ny] += em
        values /= 2 * structural_mass
        dimensions = 3

        dcg_dnodes = np.tile(values, dimensions)
        if is_sym:
            dcg_dnodes[ny : 2 * ny] = 0
            dcg_dnodes *= 2

            dcg_dmass[1] = 0
            dcg_dmass *= 2

            dcg_dem[1] = 0
            dcg_dem *= 2

        J["cg_location", "structural_mass"] = dcg_dmass
        J["cg_location", "element_mass"] = dcg_dem
        J["cg_location", "nodes"] = dcg_dnodes
