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
    coerces ``surface``/``surfaces`` to ``Surface`` in place, so the validator and the field
    see the same object.

    Parameters
    ----------
    cls : type
        The component or group class being validated.
    data : dict
        The raw constructor data; modified in place for ``surface``/``surfaces``.

    Returns
    -------
    SimpleNamespace
        One attribute per model field.
    """
    if "surface" in data and not isinstance(data["surface"], Surface):
        data["surface"] = Surface.model_validate(data["surface"])
    if "surfaces" in data:
        data["surfaces"] = [s if isinstance(s, Surface) else Surface.model_validate(s) for s in data["surfaces"]]

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
