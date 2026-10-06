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
