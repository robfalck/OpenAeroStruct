"""
B-spline interpolation component.

om4 has no ``SplineComp`` (see ``ai/OM4_NEEDS.md``), so OAS carries this
single-spline equivalent of OM3's ``SplineComp(method="bsplines")``.  The basis
matrix is a verbatim port of OM3's ``InterpBSplines.get_bspline_mtx`` so the
interpolated values match OM3 to round-off.
"""

from functools import cached_property
from typing import Any

import numpy as np
from pydantic import Field, model_validator

import om4.api as om
from om4.utils.types import FloatOrNpArray


def get_bspline_mtx(num_cp, t_vec, order=4):
    """
    Compute the (num_pt, num_cp) matrix of B-spline basis coefficients.

    Parameters
    ----------
    num_cp : int
        Number of control points.
    t_vec : ndarray
        Interpolation point locations, mapped onto [0, 1].
    order : int
        B-spline order.

    Returns
    -------
    data, rows, cols : ndarray
        COO entries of the basis matrix, ``order`` entries per interpolation point.
    """
    knots = np.zeros(num_cp + order)
    knots[order - 1 : num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1 :] = 1.0

    basis = np.zeros(order, dtype=t_vec.dtype)
    arange = np.arange(order)

    num_pt = len(t_vec)
    data = np.zeros((num_pt, order), dtype=t_vec.dtype)
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        if t.real == knots[-1].real:
            i0 = num_cp - order
        else:
            for ind in range(order, num_cp + 1):
                if (knots[ind - 1].real <= t.real) and (t.real < knots[ind].real):
                    i0 = ind - order
                    break

        basis[:] = 0.0
        basis[-1] = 1.0

        for i in range(2, order + 1):
            ll = i - 1
            j1 = order - ll
            j2 = order
            n = i0 + j1

            if knots[n + ll] != knots[n]:
                basis[j1 - 1] = (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.0

            for j in range(j1 + 1, j2):
                n = i0 + j

                if knots[n + ll - 1] != knots[n - 1]:
                    basis[j - 1] = (t - knots[n - 1]) / (knots[n + ll - 1] - knots[n - 1]) * basis[j - 1]
                else:
                    basis[j - 1] = 0.0

                if knots[n + ll] != knots[n]:
                    basis[j - 1] += (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j]

            n = i0 + j2
            if knots[n + ll - 1] != knots[n - 1]:
                basis[j2 - 1] = (t - knots[n - 1]) / (knots[n + ll - 1] - knots[n - 1]) * basis[j2 - 1]
            else:
                basis[j2 - 1] = 0.0

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    return data.ravel(), rows.ravel(), cols.ravel()


def _bspline_coo(num_cp, x_interp, order, x_cp_start, x_cp_end):
    """Map x_interp onto [0, 1] the way OM3's InterpBSplines does and return the COO basis."""
    x = np.atleast_1d(np.asarray(x_interp, dtype=float))
    start = x_cp_start if x_cp_start is not None else x[0]
    end = x_cp_end if x_cp_end is not None else x[-1]
    x_mapped = (x - min(start, end)) / (end - start)
    return get_bspline_mtx(num_cp, x_mapped, order=order)


class BsplineComp(om.ExplicitComponent):
    """
    Interpolate ``num_cp`` control points onto ``x_interp`` with a B-spline.

    The output is linear in the control points, so the single partial is a constant
    declared at construction.
    """

    num_cp: int = Field(description="Number of control points.")
    x_interp: FloatOrNpArray = Field(description="Locations of the interpolated points.")
    order: int = Field(default=4, description="B-spline order.")
    x_cp_start: float | None = Field(
        default=None, description="Location of the first control point (default: x_interp[0])."
    )
    x_cp_end: float | None = Field(
        default=None, description="Location of the last control point (default: x_interp[-1])."
    )
    cp_name: str = Field(description="Name of the control point input.")
    interp_name: str = Field(description="Name of the interpolated output.")
    units: str | None = Field(default=None, description="Units of both input and output.")
    cp_val: FloatOrNpArray | None = Field(default=None, description="Default control point values.")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        num_cp = int(data["num_cp"])
        n_interp = np.size(data["x_interp"])
        units = data.get("units")
        cp_val = data.get("cp_val")
        cp_val = np.ones(num_cp) if cp_val is None else np.asarray(cp_val, dtype=float).reshape(num_cp)

        vals, rows, cols = _bspline_coo(
            num_cp, data["x_interp"], data.get("order", 4), data.get("x_cp_start"), data.get("x_cp_end")
        )
        data["inputs"] = {data["cp_name"]: om.InputVar(val=cp_val, units=units)}
        data["outputs"] = {data["interp_name"]: om.OutputVar(val=np.ones(n_interp), units=units)}
        data["partials"] = [
            om.PartialsSpec(of=data["interp_name"], wrt=data["cp_name"], rows=rows, cols=cols, val=vals)
        ]
        return data

    @cached_property
    def _basis(self) -> np.ndarray:
        """Dense (n_interp, num_cp) basis matrix."""
        vals, rows, cols = _bspline_coo(self.num_cp, self.x_interp, self.order, self.x_cp_start, self.x_cp_end)
        mtx = np.zeros((np.size(self.x_interp), self.num_cp))
        np.add.at(mtx, (rows, cols), vals)
        return mtx

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs[self.interp_name] = self._basis @ inputs[self.cp_name]
