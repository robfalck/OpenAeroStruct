from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class LiftCoeff2D(om.ExplicitComponent):
    """
    Calculate 2D lift coefficient distribution based on section forces.
    This is for one given lifting surface. These are the sectional Cls.

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees.
    sec_forces[nx-1, ny-1, 3] : numpy array
        Flattened array containing the sectional forces acting on each panel.
        Stored in Fortran order (only relevant with more than one chordwise
        panel).
    widths[ny-1] : numpy array
        The spanwise widths of each individual panel.
    chords[ny] : numpy array
        The chordwise distance between the leading and trailing edges.
    v : float
        Freestream air velocity in m/s.
    rho : float
        Air density in kg/m^3.

    Returns
    -------
    Cl[ny-1] : numpy array
        2D lift coefficient distribution for the lifting surface.

    """

    surface: Surface

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surface = f["surface"]

        nx, ny = surface["mesh"].shape[:2]

        # Inputs
        d.add_input("alpha", val=3.0, units="deg", tags=["mphys_input"])
        d.add_input("sec_forces", val=np.ones((nx - 1, ny - 1, 3)), units="N", tags=["mphys_coupling"])
        d.add_input("widths", val=np.ones((ny - 1)) * 0.2, units="m", tags=["mphys_coupling"])
        d.add_input("chords", val=np.ones((ny)), units="m", tags=["mphys_coupling"])
        d.add_input("v", val=1.0, units="m/s", tags=["mphys_input"])
        d.add_input("rho", val=1.0, units="kg/m**3", tags=["mphys_input"])

        # Outputs
        d.add_output("Cl", val=np.zeros((ny - 1)))

        d.declare_partials("Cl", "widths")
        d.declare_partials("Cl", "v")
        d.declare_partials("Cl", "rho")
        d.declare_partials("Cl", "alpha")

        # Added to declare Jacobian sparse
        arange = np.arange(ny - 1)
        rows = np.tile(arange, 2)
        cols = np.hstack((arange, arange + 1))
        d.declare_partials("Cl", "chords", rows=rows, cols=cols)

        rows = np.tile(np.repeat(arange, 3), nx - 1)
        cols = np.arange((ny - 1) * (nx - 1) * 3)
        d.declare_partials("Cl", "sec_forces", rows=rows, cols=cols)
        return d.into(_spec)

    @property
    def nx(self) -> int:
        return self.surface["mesh"].shape[0]

    @property
    def ny(self) -> int:
        return self.surface["mesh"].shape[1]

    @property
    def num_panels(self) -> int:
        return (self.nx - 1) * (self.ny - 1)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Input parameters
        alpha = inputs["alpha"] * np.pi / 180.0
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        sec_forces = inputs["sec_forces"]
        widths = inputs["widths"]
        chords = inputs["chords"]
        v = inputs["v"]
        rho = inputs["rho"]

        # Lift distribution: dimensional l(y) = -Fx(y) sin(alpha) + Fz(y) cos(alpha) / widths(y)
        forces = np.sum(sec_forces, axis=0)  # sum section forces in the chordwise x-direction: forces(ny,3)
        lift_dist = (-forces[:, 0] * sina + forces[:, 2] * cosa) / widths[:]

        # Mid-panel chord
        chord = 0.5 * (chords[1:] + chords[:-1])  # chord c(y)

        # Lift coefficient distribution
        outputs["Cl"] = lift_dist[:] / (0.5 * rho * v**2 * chord[:])

    def compute_partials(self, inputs, partials):
        """Jacobian for 2D lift coefficient distribution."""

        # Input parameters
        alpha = inputs["alpha"] * np.pi / 180.0
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        sec_forces = inputs["sec_forces"]
        widths = inputs["widths"]
        chords = inputs["chords"]
        v = inputs["v"]
        rho = inputs["rho"]

        # Lift distribution: dimensional l(y) = -Fx(y) sin(alpha) + Fz(y) cos(alpha) / widths(y)
        forces = np.sum(sec_forces, axis=0)  # sum section forces in the chordwise x-direction: forces(ny,3)
        lift_dist = (-forces[:, 0] * sina + forces[:, 2] * cosa) / widths[:]

        # Mid-panel chord
        chord = 0.5 * (chords[1:] + chords[:-1])  # chord c(y)

        # Linearization of lift coefficient distribution
        # outputs['Cl'] = lift_dist[:] / ( 0.5 * rho * v**2 * chord[:] )

        # Analytic derivatives for alpha
        p180 = np.pi / 180.0
        partials["Cl", "alpha"] = (
            p180 * (-forces[:, 0] * cosa - forces[:, 2] * sina) / widths[:] / (0.5 * rho * v**2 * chord[:])
        )

        # Analytic derivatives for sec_forces
        tmp = np.concatenate((-sina, np.array([0]), cosa))

        #         ### Replaced to vectorize computation
        #         print(tmp)
        #         for ix in range(self.nx-1):
        #             for jy in range(self.ny-1):
        #                 for ind in range(3):
        #                    partials['Cl', 'sec_forces'][jy, ix*(self.ny-1)*3 + jy*3 + ind] = \
        #                        tmp[ind] / widths[jy] / ( 0.5 * rho * v**2 * chord[jy] )
        #
        partials["Cl", "sec_forces"] = np.ravel(
            np.tile(np.einsum("i,j,j->ji", tmp, 1 / widths, 1 / (0.5 * rho * v**2 * chord)), (self.nx - 1, 1))
        )

        # Analytic derivatives for widths
        partials["Cl", "widths"] = np.diag(
            -1.0 / widths[:] ** 2 * (-forces[:, 0] * sina + forces[:, 2] * cosa) / (0.5 * rho * v**2 * chord[:])
        )

        # Analytic derivatives for chords
        tmp_der = -1 / (0.5 * (chords[:-1] + chords[1:]) ** 2) * lift_dist / (0.5 * rho * v**2)
        partials["Cl", "chords"] = list(tmp_der) * 2

        #        ### Replaced to vectorize computation
        #         for iy in range(self.ny-1):
        #             partials['Cl', 'chords'][iy,iy  ] = \
        #                              -1. / ( 0.5 * (chords[iy] + chords[iy+1])**2 ) * \
        #                              lift_dist[iy] / ( 0.5 * rho * v**2 )
        #             partials['Cl', 'chords'][iy,iy+1] = \
        #                              -1. / ( 0.5 * (chords[iy] + chords[iy+1])**2 ) * \
        #                              lift_dist[iy] / ( 0.5 * rho * v**2 )
        #

        # Analytic derivatives for v
        partials["Cl", "v"] = -2.0 / v**3 * lift_dist[:] / (0.5 * rho * chord[:])

        # Analytic derivatives for rho
        partials["Cl", "rho"] = -1.0 / rho**2 * lift_dist[:] / (0.5 * v**2 * chord[:])
