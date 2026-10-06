"""
Cross-check an om4-ported component against its OM3 original (dev-only; needs ``openmdao``).

Given an OM3 component instance and the om4 port, set identical input values in both,
run them, and compare every output and every analytic subjacobian (dense).  This catches
transcription errors in ``compute``/``compute_partials`` and in sparsity patterns, which
om4's own ``check_partials`` cannot fully see (ai/OM4_NEEDS.md B-001, B-003).

The OM3 source of a module can be recovered with ``git show main:<path>`` into a temp
file and imported with :func:`load_om3_module`.
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_om3_module(rel_path, ref="main"):
    """Import the OM3 version of ``rel_path`` (e.g. 'openaerostruct/geometry/geometry_mesh.py') from git ``ref``."""
    src = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / ("om3_" + Path(rel_path).name)
    tmp.write_text(src)
    name = "om3_" + rel_path.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _om3_jac(prob):
    """Dense analytic subjacs of the OM3 component 'comp', keyed (of, wrt)."""
    import openmdao.api as om3  # noqa: F401

    data = prob.check_partials(out_stream=None, method="fd")
    return {key: np.asarray(val["J_fwd"]) for key, val in data["comp"].items()}


def crosscheck(om3_comp, om4_comp, input_vals, rtol=1e-12, atol=1e-12, check_partials=True):
    """
    Compare outputs and dense analytic partials of an OM3 component and its om4 port.

    Parameters
    ----------
    om3_comp : openmdao Component
        The original component.
    om4_comp : om4 Component
        The port.
    input_vals : dict
        ``{input_name: value}`` set in both models before running.

    Returns
    -------
    list[str]
        Mismatch descriptions; empty when the two agree.
    """
    import openmdao.api as om3
    import om4.api as om

    p3 = om3.Problem(reports=False)
    p3.model.add_subsystem("comp", om3_comp)
    p3.setup()
    p4 = om.Problem(model=om.Group(subsystems={"comp": om4_comp}))

    for name, val in input_vals.items():
        p3.set_val(f"comp.{name}", val)
        p4.set_val(f"comp.{name}", np.asarray(val).reshape(np.shape(p4.get_val(f"comp.{name}"))))
    p3.run_model()
    p4.run_model()

    problems = []
    for name in om4_comp.outputs:
        v3 = np.asarray(p3.get_val(f"comp.{name}")).ravel()
        v4 = np.asarray(p4.get_val(f"comp.{name}")).ravel()
        if v3.shape != v4.shape or not np.allclose(v3, v4, rtol=rtol, atol=atol):
            err = np.max(np.abs(v3 - v4)) if v3.shape == v4.shape else f"shape {v3.shape} vs {v4.shape}"
            problems.append(f"output {name}: max diff {err}")

    if check_partials:
        j3 = _om3_jac(p3)
        res4 = p4.check_partials(only_incorrect=False, atol=1e300, rtol=1e300).get("comp", {})
        j4 = {key: np.asarray(val["J"]) for key, val in res4.items()}
        for key in set(j3) | set(j4):
            a = j3.get(key)
            b = j4.get(key)
            if a is None:
                a = np.zeros_like(b)
            if b is None:
                b = np.zeros_like(a)
            a = a.reshape(b.shape) if a.size == b.size else a
            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
                err = np.max(np.abs(a - b)) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}"
                problems.append(f"partial d{key[0]}/d{key[1]}: max diff {err}")
    return problems


def om3_oas(ref="main"):
    """
    Make the complete OM3 OpenAeroStruct from git ``ref`` importable as package ``oas_om3``.

    The tree is exported to a cache directory with every ``openaerostruct`` import
    rewritten to ``oas_om3``, so OM3 groups import their OM3 children, side by side
    with the om4 port.
    """
    import io
    import re
    import tarfile

    cache = Path(tempfile.gettempdir()) / f"oas_om3_{ref.replace('/', '_')}"
    pkg = cache / "oas_om3"
    if not pkg.exists():
        tar_bytes = subprocess.run(
            ["git", "archive", ref, "openaerostruct"], cwd=REPO, check=True, capture_output=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            tf.extractall(cache, filter="data")
        (cache / "openaerostruct").rename(pkg)
        for py in pkg.rglob("*.py"):
            py.write_text(re.sub(r"\bopenaerostruct\b", "oas_om3", py.read_text(encoding="utf-8")), encoding="utf-8")
    if str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    import oas_om3

    return oas_om3


def random_inputs(om4_comp, overrides=None, seed=0):
    """Random input values shaped like ``om4_comp``'s inputs, with ``overrides`` taking precedence."""
    rng = np.random.default_rng(seed)
    vals = {}
    for name, var in om4_comp.inputs.items():
        shape = var.shape if var.shape is not None else np.shape(var.val)
        vals[name] = rng.random(shape) + 0.5
    vals.update(overrides or {})
    return vals


def report(label, problems):
    """Print one line per crosscheck and return 1 on failure (for counting)."""
    print(f"{label:55} {'OK' if not problems else problems}")
    return int(bool(problems))
