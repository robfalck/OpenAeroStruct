import unittest

import numpy as np

import om4.api as om

from openaerostruct.aerodynamics.wave_drag import WaveDrag
from openaerostruct.utils.bspline_comp import BsplineComp
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surface = get_default_surfaces()[0]
        surface["with_wave"] = True
        surface["t_over_c_cp"] = np.array([0.15, 0.21, 0.03, 0.05])

        ny = surface["mesh"].shape[1]
        n_cp = len(surface["t_over_c_cp"])

        x_interp = np.linspace(0.0, 1.0, int(ny - 1))
        group = om.Group(
            subsystems={
                "t_over_c_bsp": BsplineComp(
                    num_cp=n_cp,
                    x_interp=x_interp,
                    order=min(n_cp, 4),
                    cp_name="t_over_c_cp",
                    interp_name="t_over_c",
                    cp_val=surface["t_over_c_cp"],
                    promotes_inputs=["t_over_c_cp"],
                    promotes_outputs=["t_over_c"],
                ),
                "wavedrag": WaveDrag(
                    surface=surface,
                    promotes_inputs=["t_over_c", "Mach_number", "CL", "lengths_spanwise", "widths", "chords"],
                ),
            }
        )

        values = {
            "comp.Mach_number": 0.95,
            "comp.CL": 0.7,
            "comp.lengths_spanwise": np.array([12.14757848, 11.91832712, 11.43730892]),
            "comp.widths": np.array([10.01555924, 9.80832351, 9.79003729]),
            "comp.chords": np.array([2.72835132, 5.12528179, 7.88916016, 13.6189974]),
        }

        run_test(self, group, complex_flag=True, values=values)


if __name__ == "__main__":
    unittest.main()
