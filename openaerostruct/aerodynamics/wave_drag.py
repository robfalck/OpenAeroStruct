from typing import Any, ClassVar

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import PartialsBuffer, VarDecl, field_values
from openaerostruct.utils.surface import Surface


class WaveDrag(om.ExplicitComponent):
    """
    Compute the wave drag if the with_wave option is True. If not, the CDw is 0.
    This component exists for each lifting surface.

    Parameters
    ----------
    Mach_number : float
        Mach number.
    widths[ny-1] : numpy array
        The width in the spanwise direction of each VLM panel. This is the numerator of cos(sweep).
    lengths_spanwise[ny-1] : numpy array
        The spanwise length of each VLM panel at 1/4 chord, rotated by the sweep angle. This is the denominator
        of cos(sweep)
    CL : float
        The CL of the lifting surface used for wave drag estimation.
    chords[ny] : numpy array
        The chord length of each mesh slice. This is dimension ny rather than ny-1 which would be
        expected for chord length of each VLM panel.
    t_over_c[ny-1] : numpy array
        The streamwise thickness-to-chord ratio of each VLM panel.

    Returns
    -------
    CDw : float
        Wave drag coefficient for the lifting surface computed using equations based on the
        Korn equation
    """

    surface: Surface
    with_wave: bool | None = None  # unused; the surface's with_wave decides

    ka: ClassVar[float] = 0.95  # airfoil technology level (for NASA SC airfoil)

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surface = f["surface"]
        # As in OM3, the surface decides; the with_wave field is overwritten.
        _spec["with_wave"] = surface["with_wave"]

        # Thickness over chord for the airfoil

        ny = surface["mesh"].shape[1]

        d.add_input("Mach_number", val=1.6, tags=["mphys_input"])
        d.add_input("widths", val=np.ones((ny - 1)) * 0.2, units="m", tags=["mphys_coupling"])
        d.add_input(
            "lengths_spanwise", val=np.arange((ny - 1)) + 1.0, units="m", tags=["mphys_coupling"]
        )  # set to np.arange so that d_CDw_d_chords is nonzero
        d.add_input("CL", val=0.33)
        d.add_input("chords", val=np.ones((ny)), units="m", tags=["mphys_coupling"])
        d.add_input("t_over_c", val=np.arange((ny - 1)), tags=["mphys_input"])
        d.add_output("CDw", val=0.0)

        d.declare_partials("CDw", "*")
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.with_wave:
            t_over_c = inputs["t_over_c"]
            widths = inputs["widths"]
            cos_sweep = widths / inputs["lengths_spanwise"]
            M = inputs["Mach_number"]
            chords = inputs["chords"]
            CL = inputs["CL"]

            panel_mid_chords = (chords[:-1] + chords[1:]) / 2.0
            panel_areas = panel_mid_chords * widths
            sum_panel_areas = np.sum(panel_areas)
            avg_cos_sweep = np.sum(cos_sweep * panel_areas) / sum_panel_areas  # weighted average of 1/4 chord sweep
            avg_t_over_c = np.sum(t_over_c * panel_areas) / sum_panel_areas  # weighted average of streamwise t/c
            MDD = self.ka / avg_cos_sweep - avg_t_over_c / avg_cos_sweep**2 - CL / (10 * avg_cos_sweep**3)
            Mcrit = MDD - (0.1 / 80.0) ** (1.0 / 3.0)

            if np.real(M) > np.real(Mcrit):
                outputs["CDw"] = 20 * (M - Mcrit) ** 4
            else:
                outputs["CDw"] = 0.0

            if self.surface["symmetry"]:
                outputs["CDw"] *= 2
        else:
            outputs["CDw"] = 0.0

    def compute_partials(self, inputs, partials):
        """Jacobian for wave drag."""
        # Explicitly zero out the partials to begin, we can't assume the input partials arrays contain zeros already
        # om4 subjacs are write-only (ai/OM4_NEEDS.md N-005); fill locally and flush once
        partials_out, partials = partials, PartialsBuffer(self)
        partials["CDw", "CL"][:] = 0.0
        partials["CDw", "lengths_spanwise"][:] = 0.0
        partials["CDw", "widths"][:] = 0.0
        partials["CDw", "Mach_number"][:] = 0.0
        partials["CDw", "chords"][:] = 0.0
        partials["CDw", "t_over_c"][:] = 0.0

        if self.with_wave:
            ny = self.surface["mesh"].shape[1]
            t_over_c = inputs["t_over_c"]
            widths = inputs["widths"]
            lengths_spanwise = inputs["lengths_spanwise"]
            cos_sweep = widths / lengths_spanwise
            M = inputs["Mach_number"]
            chords = inputs["chords"]
            CL = inputs["CL"]

            panel_mid_chords = (chords[:-1] + chords[1:]) / 2.0
            panel_areas = panel_mid_chords * widths
            sum_panel_areas = np.sum(panel_areas)
            avg_cos_sweep = np.sum(cos_sweep * panel_areas) / sum_panel_areas
            avg_t_over_c = np.sum(t_over_c * panel_areas) / sum_panel_areas

            MDD = self.ka / avg_cos_sweep - avg_t_over_c / avg_cos_sweep**2 - CL / (10 * avg_cos_sweep**3)
            Mcrit = MDD - (0.1 / 80.0) ** (1.0 / 3.0)

            if np.real(M) > np.real(Mcrit):
                dCDwdMDD = -80 * (M - Mcrit) ** 3
                dMDDdCL = -1.0 / (10 * avg_cos_sweep**3)
                dMDDdavg = (-10 * self.ka * avg_cos_sweep**2 + 20 * avg_t_over_c * avg_cos_sweep + 3 * CL) / (
                    10 * avg_cos_sweep**4
                )
                dMDDdtoc = -1.0 / (avg_cos_sweep**2)
                dtocavgdtoc = panel_areas / sum_panel_areas

                ccos = np.sum(widths * panel_mid_chords)
                ccos2w = np.sum(panel_mid_chords * widths**2 / lengths_spanwise)

                davgdcos = 2 * panel_mid_chords * widths / lengths_spanwise / ccos - panel_mid_chords * ccos2w / ccos**2
                dtocdcos = (
                    panel_mid_chords * t_over_c / ccos
                    - panel_mid_chords * np.sum(panel_mid_chords * widths * t_over_c) / ccos**2
                )
                davgdw = -1 * panel_mid_chords * widths**2 / lengths_spanwise**2 / ccos
                davgdc = widths**2 / lengths_spanwise / ccos - widths * ccos2w / ccos**2
                dtocdc = t_over_c * widths / ccos - widths * np.sum(panel_mid_chords * widths * t_over_c) / ccos**2

                dcdchords = np.zeros((ny - 1, ny))
                i, j = np.indices(dcdchords.shape)
                dcdchords[i == j] = 0.5
                dcdchords[i == j - 1] = 0.5

                partials["CDw", "Mach_number"] = -1 * dCDwdMDD
                partials["CDw", "CL"] = dCDwdMDD * dMDDdCL
                partials["CDw", "lengths_spanwise"] = dCDwdMDD * dMDDdavg * davgdw
                partials["CDw", "widths"] = dCDwdMDD * dMDDdavg * davgdcos + dCDwdMDD * dMDDdtoc * dtocdcos
                partials["CDw", "chords"] = dCDwdMDD * dMDDdavg * np.matmul(
                    davgdc, dcdchords
                ) + dCDwdMDD * dMDDdtoc * np.matmul(dtocdc, dcdchords)
                partials["CDw", "t_over_c"] = dCDwdMDD * dMDDdtoc * dtocavgdtoc

        if self.surface["symmetry"]:
            partials["CDw", "CL"][0, :] *= 2
            partials["CDw", "lengths_spanwise"][0, :] *= 2
            partials["CDw", "widths"][0, :] *= 2
            partials["CDw", "Mach_number"][0, :] *= 2
            partials["CDw", "chords"][0, :] *= 2
            partials["CDw", "t_over_c"][0, :] *= 2

        partials.flush(partials_out)
