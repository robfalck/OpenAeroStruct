"""
Mechanical first pass of an OM3 OAS component module onto om4 (dev tool).

Rewrites, per class:
- ``initialize`` (options)                -> typed fields (given by the caller)
- ``setup`` (add_input/add_output/...)    -> a ``@model_validator(mode="before")`` building a VarDecl
- ``self.options["x"]``                   -> ``self.x`` (runtime) / ``f["x"]`` (validator)
- ``compute``                             -> ``compute_outputs``
- ``apply_nonlinear``/``linearize``       -> ``compute_residuals``/``compute_partials`` (implicit)
- ``self.system_size = ...`` in setup     -> dropped; a ``system_size`` cached_property is added

It does NOT handle subjac slice writes (ai/OM4_NEEDS.md N-005), ``method="cs"`` partials
(N-002), or other instance state; the caller fixes those by hand and must cross-check the
result against OM3 (ai/tools/om3_crosscheck.py).
"""

import re

IMPORTS = """from functools import cached_property
from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface
"""

SYSTEM_SIZE = '''
    @cached_property
    def system_size(self) -> int:
        """Total number of VLM panels over all surfaces."""
        return vlm_system_size(self.surfaces)
'''


def port_class(cls_src, fields, implicit=False):
    """Rewrite one class body (text from ``class X`` to the next top-level statement)."""
    if "    def initialize(self):" in cls_src:
        head, rest = cls_src.split("    def initialize(self):", 1)
        _, rest = rest.split("    def setup(self):", 1)
    else:
        head, rest = cls_src.split("    def setup(self):", 1)
    compute_sig = "    def apply_nonlinear(self, inputs, outputs, residuals):" if implicit else None
    if implicit:
        setup_body, rest = rest.split(compute_sig, 1)
        rest = compute_sig + rest
    else:
        setup_body, rest = re.split(r"    def compute\(self, inputs, outputs\):", rest, maxsplit=1)
        rest = "    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):" + rest

    uses_system_size = "self.system_size" in setup_body or "self.system_size" in rest
    setup_body = re.sub(r"^\s*self\.system_size = system_size\n", "", setup_body, flags=re.M)
    setup_body = re.sub(r"self\.options\[", "f[", setup_body)
    setup_body = re.sub(r"self\.(add_input|add_output|declare_partials)\(", r"d.\1(", setup_body)
    setup_body = re.sub(r"^\s*self\.set_check_partial_options\(.*\)\n", "", setup_body, flags=re.M)

    rest = re.sub(r'self\.options\["(\w+)"\]', r"self.\1", rest)
    if implicit:
        rest = rest.replace(
            "    def apply_nonlinear(self, inputs, outputs, residuals):",
            "    def compute_residuals(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):",
        )
        rest = rest.replace("    def linearize(self, inputs, outputs, partials):", "    def compute_partials(self, inputs, outputs, partials):")

    validator = f"""{fields}

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`
{setup_body.rstrip()}
        return d.into(_spec)
{SYSTEM_SIZE if uses_system_size else ""}
"""
    return head + validator + rest


def port_module(path, class_fields, implicit=()):
    """Port every class in ``class_fields`` ({class name: fields source}) in the module at ``path``."""
    src = open(path).read()
    src = re.sub(r"import openmdao\.api as om\n", "", src, count=1)
    src = src.replace("import numpy as np\n", IMPORTS, 1)
    parts = re.split(r"(?m)^(?=class \w+)", src)
    out = []
    for part in parts:
        m = re.match(r"class (\w+)", part)
        if m and m.group(1) in class_fields:
            part = port_class(part, class_fields[m.group(1)], implicit=m.group(1) in implicit)
        out.append(part)
    open(path, "w").write("".join(out))
