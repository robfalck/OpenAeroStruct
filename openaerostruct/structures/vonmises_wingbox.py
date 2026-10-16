from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives.complex_step import ComplexStep
from om4.derivatives.differentiator import Differentiator
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.utils import norm, unit
from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class VonMisesWingbox(om.ExplicitComponent):
    """Compute the von Mises stresses for each element.
    See Chauhan et al. (https://doi.org/10.1007/978-3-319-97773-7_38) for more.

    Parameters
    ----------
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.
    disp[ny, 6] : numpy array
        Displacements of each FEM node.
    Qz[ny-1] : numpy array
        First moment of area above the neutral axis parallel to the local
        z-axis (for each wingbox segment).
    J[ny-1] : numpy array
        Torsion constants for each wingbox segment.
    A_enc[ny-1] : numpy array
        Cross-sectional enclosed area (measured using the material midlines) of
        each wingbox segment.
    spar_thickness[ny-1] : numpy array
        Material thicknesses of the front and rear spars for each wingbox segment.
    htop[ny-1] : numpy array
        Distance to the point on the top skin that is the farthest away from
        the local-z neutral axis (for each wingbox segment).
    hbottom[ny-1] : numpy array
        Distance to the point on the bottom skin that is the farthest away from
        the local-z neutral axis (for each wingbox segment).
    hfront[ny-1] : numpy array
        Distance to the point on the front spar that is the farthest away from
        the local-y neutral axis (for each wingbox segment).
    hrear[ny-1] : numpy array
        Distance to the point on the rear spar that is the farthest away
        from the local-y neutral axis (for each wingbox segment).

    Returns
    -------
    vonmises[ny-1, 4] : numpy array
        von Mises stresses for 4 stress combinations for each FEM element.

    """

    surface: Surface
    differentiator: Polymorphic[Differentiator] | None = ComplexStep()

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        ny = f.surface["mesh"].shape[1]
        d = VarDecl()

        d.add_input("nodes", val=np.zeros((ny, 3)), units="m")
        d.add_input("disp", val=np.zeros((ny, 6)), units="m")
        d.add_input("Qz", val=np.zeros((ny - 1)), units="m**3")
        d.add_input("J", val=np.zeros((ny - 1)), units="m**4")
        d.add_input("A_enc", val=np.zeros((ny - 1)), units="m**2")
        d.add_input("spar_thickness", val=np.zeros((ny - 1)), units="m")
        d.add_input("htop", val=np.zeros((ny - 1)), units="m")
        d.add_input("hbottom", val=np.zeros((ny - 1)), units="m")
        d.add_input("hfront", val=np.zeros((ny - 1)), units="m")
        d.add_input("hrear", val=np.zeros((ny - 1)), units="m")

        d.add_output("vonmises", val=np.zeros((ny - 1, 4)), units="N/m**2")

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        disp = inputs["disp"]
        nodes = inputs["nodes"]
        A_enc = inputs["A_enc"]
        Qy = inputs["Qz"]
        J = inputs["J"]
        htop = inputs["htop"]
        hbottom = inputs["hbottom"]
        hfront = inputs["hfront"]
        hrear = inputs["hrear"]
        spar_thickness = inputs["spar_thickness"]

        dtype = np.result_type(disp, nodes, A_enc, Qy, J, htop, hbottom, hfront, hrear, spar_thickness)
        T = np.zeros((3, 3), dtype=dtype)
        x_gl = np.array([1, 0, 0], dtype=dtype)
        vonmises = np.zeros((nodes.shape[0] - 1, 4), dtype=dtype)

        E = self.surface["E"]
        G = self.surface["G"]
        tssf = self.surface["strength_factor_for_upper_skin"]

        num_elems = nodes.shape[0] - 1
        for ielem in range(num_elems):
            P0 = nodes[ielem, :]
            P1 = nodes[ielem + 1, :]
            L = norm(P1 - P0)

            x_loc = unit(P1 - P0)
            y_loc = unit(np.cross(x_loc, x_gl))
            z_loc = unit(np.cross(x_loc, y_loc))

            T[0, :] = x_loc
            T[1, :] = y_loc
            T[2, :] = z_loc

            u0x, u0y, u0z = T.dot(disp[ielem, :3])
            r0x, r0y, r0z = T.dot(disp[ielem, 3:])
            u1x, u1y, u1z = T.dot(disp[ielem + 1, :3])
            r1x, r1y, r1z = T.dot(disp[ielem + 1, 3:])

            # this is stress = modulus * strain; positive is tensile
            axial_stress = E * (u1x - u0x) / L

            # this is Torque / (2 * thickness_min * Area_enclosed)
            torsion_stress = G * J[ielem] / L * (r1x - r0x) / 2 / spar_thickness[ielem] / A_enc[ielem]

            # this is moment * h / I
            top_bending_stress = E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L) * htop[ielem]

            # this is moment * h / I
            bottom_bending_stress = -E / (L**2) * (6 * u0y + 2 * r0z * L - 6 * u1y + 4 * r1z * L) * hbottom[ielem]

            # this is moment * h / I
            front_bending_stress = -E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L) * hfront[ielem]

            # this is moment * h / I
            rear_bending_stress = E / (L**2) * (-6 * u0z + 2 * r0y * L + 6 * u1z + 4 * r1y * L) * hrear[ielem]

            # shear due to bending (VQ/It) note: the I used to get V cancels the other I
            vertical_shear = (
                E
                / (L**3)
                * (-12 * u0y - 6 * r0z * L + 12 * u1y - 6 * r1z * L)
                * Qy[ielem]
                / (2 * spar_thickness[ielem])
            )

            # The 4 stress combinations:
            vonmises[ielem, 0] = (
                np.sqrt((top_bending_stress + rear_bending_stress + axial_stress) ** 2 + 3 * torsion_stress**2) / tssf
            )
            vonmises[ielem, 1] = np.sqrt(
                (bottom_bending_stress + front_bending_stress + axial_stress) ** 2 + 3 * torsion_stress**2
            )
            vonmises[ielem, 2] = np.sqrt(
                (front_bending_stress + axial_stress) ** 2 + 3 * (torsion_stress - vertical_shear) ** 2
            )
            vonmises[ielem, 3] = (
                np.sqrt((rear_bending_stress + axial_stress) ** 2 + 3 * (torsion_stress + vertical_shear) ** 2) / tssf
            )

        outputs["vonmises"] = vonmises
