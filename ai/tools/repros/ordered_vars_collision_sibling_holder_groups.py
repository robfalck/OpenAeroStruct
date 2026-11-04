"""Repro: under real multi-rank MPI, Problem construction raises a bogus
"Name '...' already exists in OrderedVars." RuntimeError for two unrelated sibling Groups that
don't actually share any variable name.

Found while porting tests/integration_tests/test_multipoint_parallel.py (a parallel multipoint
aerostructural test): a real OpenAeroStruct `AerostructGeometry("wing")` built next to an
unrelated `parallel=True` sibling failed the same way under `mpirun -n 2`. Bisection removed
every OAS-specific and `parallel=True`-specific detail; what's left is the minimal trigger below.

Trigger: a parent Group has >= 2 sibling sub-Groups, each wrapping a single leaf component that
promotes one input up to its own Group's boundary (the "holder" pattern OAS's
spline_or_broadcast uses for every control-point DV -- see openaerostruct/utils/om4_utils.py).
The two leaves' own (unpromoted) variable names differ (`_twist_cp_1d` vs `_t_over_c_cp_1d`
below) -- this is not a same-name collision. It only reproduces under genuine multi-rank MPI
(`mpirun -n 2`; does NOT reproduce under `mpirun -n 1`, even with OPENMDAO_USE_MPI=1 forcing MPI
object availability, nor under plain `python` with no launcher at all).

Neither "two sibling Groups" nor "a leaf with a promoted input" alone is enough:
  - A single holder-sibling Group builds fine (see `single_holder`).
  - Two sibling *leaf* components (no wrapping Group, each promoting its own input directly)
    also build fine (see `two_plain_leaves`) -- so the extra Group-wrapping level matters.
  - A `parallel=True` sibling is NOT required to reproduce it (contrary to an earlier, less
    isolated hypothesis) -- `two_holders` fails with no `parallel` Group anywhere in the model.

The error always reports the FIRST sibling's own (already-added) var path again while merging
the SECOND, unrelated sibling's vars in Problem._setup() (om4/core/problem.py) ->
GraphExecutionNode._setup_ordered_vars() (om4/core/graphexecnode.py) ->
OrderedVars.update_from_child() (om4/utils/ordered_vars.py) -- i.e. merging the second sibling's
subtree somehow re-presents the first sibling's path instead of (or in addition to) its own.

Run: pixi run mpirun -n 2 python ai/tools/repros/ordered_vars_collision_parallel_sibling.py
(Must use an actual 2-rank mpirun launch; `-n 1` does not trigger it.)
"""

import numpy as np

import om4.api as om


def make_holder_group(control_name: str, n_cp: int = 4) -> om.Group:
    """Mirrors the OAS spline_or_broadcast "holder" pattern: a leaf ExecComp whose single
    input is promoted up to the wrapping Group's boundary, with no outside source."""
    holder = om.ExecComp(
        exprs=[f"_{control_name}_2d = _{control_name}_1d"],
        inputs={f"_{control_name}_1d": {"val": np.ones(n_cp)}},
        outputs={f"_{control_name}_2d": {"shape": (n_cp,)}},
        promotes_inputs=[(f"_{control_name}_1d", control_name)],
    )
    return om.Group(subsystems={"holder": holder}, promotes_inputs=[control_name])


def build(label: str, wing_subsystems: dict) -> None:
    wing = om.Group(subsystems=wing_subsystems, promotes_inputs=["*"])
    try:
        om.Problem(model=om.Group(subsystems={"wing": wing}))
        print(f"{label}: OK built without error")
    except RuntimeError as exc:
        print(f"{label}: BUG -- {exc}")


# Single holder-sibling: builds fine.
build("single_holder", {"twist_bsp": make_holder_group("twist_cp")})

# Two sibling leaf ExecComps, no wrapping Group: also builds fine.
c1 = om.ExecComp(exprs=["b = 2*a"], promotes_inputs=["a"])
c2 = om.ExecComp(exprs=["d = 2*c"], promotes_inputs=["c"])
build("two_plain_leaves", {"comp1": c1, "comp2": c2})

# Two sibling holder-Groups: BUG, under real 2-rank MPI.
build(
    "two_holders",
    {"twist_bsp": make_holder_group("twist_cp"), "t_over_c_bsp": make_holder_group("t_over_c_cp")},
)
