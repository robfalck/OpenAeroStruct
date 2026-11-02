"""Helpers shared by OAS components and groups on om4."""

from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic_core import PydanticUndefined

from openaerostruct.utils.surface import Surface


def field_values(cls, data: dict[str, Any]) -> SimpleNamespace:
    """
    Return the raw field values a before-validator is about to validate, with class defaults filled in.

    Before-validators build ``inputs``/``outputs``/``partials`` from the semantic fields, but
    run before field validation, so ``data`` only holds what the caller passed.  This fills
    the gaps from the field defaults (so defaults are not repeated in the validator) and
    coerces ``surface``/``surfaces``/``sections`` to ``Surface`` in place, so the validator and the field
    see the same object.

    Parameters
    ----------
    cls : type
        The component or group class being validated.
    data : dict
        The raw constructor data; modified in place for ``surface``/``surfaces``/``sections``.

    Returns
    -------
    SimpleNamespace
        One attribute per model field.
    """
    if "surface" in data and not isinstance(data["surface"], Surface):
        data["surface"] = Surface.model_validate(data["surface"])
    for key in ("surfaces", "sections"):
        if key in data:
            data[key] = [s if isinstance(s, Surface) else Surface.model_validate(s) for s in data[key]]

    vals = {}
    for name, field in cls.model_fields.items():
        if name in data:
            vals[name] = data[name]
        elif field.default is not PydanticUndefined:
            vals[name] = field.default
        elif field.default_factory is not None:
            vals[name] = field.default_factory()
        else:
            vals[name] = None
    return SimpleNamespace(**vals)


def idx(a) -> np.ndarray:
    """Return ``a`` as an int index array (OM3 accepted float rows/cols; om4's IndexArray does not)."""
    return np.asarray(a).astype(int)


class VarDecl:
    """
    Accumulates ``inputs``/``outputs``/``partials`` inside a component's before-validator.

    Mirrors OM3's ``add_input``/``add_output``/``declare_partials`` argument semantics so the
    OM3 declaration code ports line for line, then :meth:`into` writes the result into the
    validator's ``data``.  It is only a construction helper: components still have typed
    fields, and nothing here is reachable at run time.

    OM3 semantics kept: a scalar ``val`` with no ``shape`` is shape ``(1,)``; ``units``
    defaults to ``None`` (explicitly dimensionless, rather than om4's dynamic UNSET); float
    ``rows``/``cols`` are cast to int; ``tags`` lists become sets.
    """

    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.partials = []

    @staticmethod
    def _val(val, shape):
        if shape is not None:
            shape = (shape,) if np.ndim(shape) == 0 else tuple(shape)
            return np.broadcast_to(np.asarray(val, dtype=float), shape).copy()
        arr = np.asarray(val, dtype=float)
        return arr.reshape(1) if arr.ndim == 0 else arr.copy()

    def add_input(self, name, val=1.0, shape=None, units=None, tags=None, desc=""):
        from om4.api import InputVar

        self.inputs[name] = InputVar(val=self._val(val, shape), units=units, tags=set(tags or ()), desc=desc)

    def add_output(self, name, val=1.0, shape=None, units=None, tags=None, desc="", lower=None, upper=None):
        from om4.api import OutputVar

        kw = {}
        if lower is not None:
            kw["lower"] = lower
        if upper is not None:
            kw["upper"] = upper
        self.outputs[name] = OutputVar(val=self._val(val, shape), units=units, tags=set(tags or ()), desc=desc, **kw)

    def declare_partials(self, of, wrt, rows=None, cols=None, val=None):
        from om4.api import PartialsSpec

        kw = {}
        if rows is not None:
            kw["rows"] = idx(rows)
            kw["cols"] = idx(cols)
        if val is not None:
            kw["val"] = np.asarray(val, dtype=float)
        if isinstance(of, (list, tuple)):
            of = list(of)
        if isinstance(wrt, tuple):
            wrt = list(wrt)
        self.partials.append(PartialsSpec(of=of, wrt=wrt, **kw))

    def into(self, data: dict) -> dict:
        data["inputs"] = self.inputs
        data["outputs"] = self.outputs
        data["partials"] = self.partials
        return data


def spline_or_broadcast(
    *,
    n_cp: int,
    x_interp: np.ndarray,
    control_name: str,
    interp_name: str,
    units: str | None,
    control_val: np.ndarray | None,
    x_start: float | None = None,
    x_end: float | None = None,
    promotes_inputs: list[str],
    promotes_outputs: list[str],
):
    """
    Interpolate ``control_name`` onto ``x_interp`` with a B-spline, or broadcast it.

    om4's ``SplineComp``/``BSplineCurve`` requires ``num_control >= 2`` (ai/OM4_NEEDS.md
    N-011), unlike OM3's ``SplineComp(method="bsplines")`` which degrades a single control
    point to a constant (``order=1``). OAS surfaces commonly define single-element
    control-point arrays (e.g. ``t_over_c_cp``), so that case is handled here by broadcasting
    the lone value across ``x_interp`` instead of spline-interpolating it.

    om4's ``SplineComp`` also always declares its control-point input with shape
    ``(vec_size, num_control)`` = ``(1, n_cp)`` (same N-011 entry), unlike OM3/the old local
    ``BsplineComp``, whose control-point input was 1-D. Nothing else pins that shape down to
    ``(n_cp,)`` at the group boundary (there is no sibling leaf expecting a 1-D input, unlike
    the interpolated *output*, which a downstream consumer such as ``GeometryMesh`` always
    declares 1-D). Left alone this surfaces as a 2-D design variable/`set_val`/`get_val`
    wherever ``control_name`` is used directly. So for ``n_cp >= 2`` the control value is held
    by a tiny 1-D passthrough ``ExecComp`` promoted under ``control_name``, whose output feeds
    the spline's 2-D input through an explicit connection (same-size reshape, as for the
    output shape).
    """
    import om4.api as om
    from om4.core.group import Connection

    n_interp = len(x_interp)
    default_val = np.broadcast_to(np.atleast_1d(control_val) if control_val is not None else 1.0, (n_cp,)).copy()

    if n_cp == 1:
        return om.ExecComp(
            exprs=[f"{interp_name} = {control_name} * ones({n_interp})"],
            inputs={control_name: {"units": units, "val": default_val}},
            outputs={interp_name: {"units": units, "shape": (n_interp,)}},
            promotes_inputs=promotes_inputs,
            promotes_outputs=promotes_outputs,
        )

    holder_in = f"_{control_name}_1d"
    holder_out = f"_{control_name}_2d"
    holder = om.ExecComp(
        exprs=[f"{holder_out} = {holder_in}"],
        inputs={holder_in: {"units": units, "val": default_val}},
        outputs={holder_out: {"units": units, "shape": (n_cp,)}},
        promotes_inputs=[(holder_in, control_name)],
    )
    spline = om.SplineComp(
        curve=om.BSplineCurve(num_control=n_cp, order=min(n_cp, 4), x_start=x_start, x_end=x_end),
        x_interp=x_interp,
        splines=[om.SplineSpec(control_name=holder_out, interp_name=interp_name, units=units)],
        promotes_outputs=[interp_name],
    )
    return om.Group(
        subsystems={"holder": holder, "spline": spline},
        connections=[Connection(src=f"holder.{holder_out}", tgt=f"spline.{holder_out}")],
        promotes_inputs=promotes_inputs,
        promotes_outputs=promotes_outputs,
    )


def vlm_system_size(surfaces) -> int:
    """Total number of VLM panels, sum of (nx - 1) * (ny - 1) over the surfaces."""
    return sum((s["mesh"].shape[0] - 1) * (s["mesh"].shape[1] - 1) for s in surfaces)


class PartialsBuffer(dict):
    """
    Readable, sliceable local stand-in for om4's write-only subjacs (ai/OM4_NEEDS.md N-005).

    OM3 code fills subjacs piecewise (``partials[k][:n] = a``, ``partials[k][0, 1:] += b``,
    ``partials[k] *= 2``).  om4 subjacs only accept whole assignment, so ported
    ``compute_partials`` methods write into this buffer exactly as the OM3 code wrote into
    ``partials`` and then call :meth:`flush` once.

    Each declared ``(of, wrt)`` starts as OM3's subjac did: its declared constant ``val``
    (broadcast), else zeros, shaped ``(nnz,)`` for sparse declarations and
    ``(of_size, wrt_size)`` for dense ones.  Only keys that were read or written are flushed,
    so constant subjacs the code never touches stay constant.
    """

    def __init__(self, comp):
        super().__init__()
        self._dirty = set()
        outs = {n: int(np.prod(v.shape)) for n, v in comp.outputs.items()}
        ins = {n: int(np.prod(v.shape)) for n, v in comp.inputs.items()}
        wrt_space = dict(ins)
        if comp.is_implicit():
            wrt_space.update(outs)
        for spec in comp.partials:
            for of in _match(spec.of, outs):
                wrts = [spec.wrt] if isinstance(spec.wrt, str) else list(spec.wrt)
                for wrt in (w for pat in wrts for w in _match(pat, wrt_space)):
                    if spec.rows is not None:
                        shape = (len(spec.rows),)
                    elif spec.diagonal:
                        shape = (outs[of],)
                    else:
                        shape = (outs[of], wrt_space[wrt])
                    val = np.zeros(shape) if spec.val is None else np.broadcast_to(spec.val, shape).astype(float)
                    super().__setitem__((of, wrt), np.array(val))

    def __getitem__(self, key):
        self._dirty.add(key)
        return super().__getitem__(key)

    def __setitem__(self, key, val):
        if key not in self:
            raise KeyError(f"Partial {key} was not declared.")
        self._dirty.add(key)
        buf = super().__getitem__(key)
        buf[...] = np.asarray(val).reshape(buf.shape)

    def flush(self, partials):
        """Assign every touched subjac to the om4 ``partials`` once."""
        for key in self._dirty:
            partials[key] = super().__getitem__(key)


def _match(pattern, names):
    import fnmatch

    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]
