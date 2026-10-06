# Architecture: OpenAeroStruct on om4

Living document for the `om4` branch. Read `../om4.git/ai/ARCHITECTURE.md` first. The
porting conventions follow dymos4 (`../dymos4.git/ai/ARCHITECTURE.md`). Plan:
`ai/OM4_MIGRATION_PLAN.md`. om4 gaps: `ai/OM4_NEEDS.md`.

## Porting rules

1. **Port the framework shell, keep the math.** The bodies of `compute`/`compute_partials`
   stay as close to the OM3 code as possible, so the numerical diff is reviewable.
2. **No `options`.** Every OM3 option is a typed Pydantic field in the class body. There is
   no `self.options` view or `add_input`/`declare_partials` compatibility shim.
3. **Components build their variables in a before-validator.** Variables and partials are
   built in a `@model_validator(mode="before")` that fills `data["inputs"]`,
   `data["outputs"]`, `data["partials"]`, following the `om4/components/ks_comp.py` pattern.
   Avoid `model_post_init`. Values derived from fields that `compute_*` needs are
   `functools.cached_property`s, or are recomputed.
4. **Groups** follow the dymos4 pattern: a `_<thing>_kwargs(...)` builder returns
   `subsystems`/`connections`/`input_defaults`/solvers, and the `Group` subclass's
   before-validator calls it.
5. **Shapes match OM3.** OM3 scalars are shape `(1,)`, so ported variables declare `(1,)`
   explicitly. Units are always given explicitly, `None` for dimensionless, because an
   om4 input with `units` UNSET becomes dynamic `by_conn`.
6. **Renames:** `compute` → `compute_outputs`; implicit `apply_nonlinear` →
   `compute_residuals`; `linearize` → `compute_partials`.
7. **Subjacs are write-only** (`ai/OM4_NEEDS.md` N-005). Piecewise fills
   (`partials[k][:n] = a; partials[k][n:] += b`) become a local array that is assigned once.
8. **Construction and run-time helpers** (`openaerostruct/utils/om4_utils.py`):
   `field_values` (field values with defaults inside a before-validator), `VarDecl` (OM3
   `add_input`/`add_output`/`declare_partials` argument semantics for building the
   validator's `inputs`/`outputs`/`partials`), and `PartialsBuffer` (a sliceable local
   subjac store flushed to om4 once, for OM3 code that fills subjacs piecewise).
9. **No run-time state set in `compute` and read in `compute_partials`.** Recompute it, or
   factor it into a helper both call (TD-105).
10. **Every port is cross-checked against OM3** with `ai/tools/om3_crosscheck.py`
    (outputs plus dense analytic Jacobians, and totals for groups) before it is committed.
11. Tests keep their OM3 reference values. Partials tests go through
   `openaerostruct/utils/testing.py:run_test`.

## Decisions (ADL)

| ID | Decision | Why / rejected |
|---|---|---|
| ADL-001 | The `surface` dict becomes `openaerostruct.utils.surface.Surface` (typed, strict). Legacy dicts are validated into it. It keeps a read-only mapping protocol (`s["mesh"]`, `"twist_cp" in s`, `s.get(k, d)`) so component math is unchanged. Unknown keys in a legacy dict warn and are dropped, as `check_surface_dict_keys` did | om4 fields must be strict and serializable. Rejected: `dict[str, Any]` (no validation, fights `strict=True`); `extra="forbid"` on dicts (breaks fixtures that carry mesh-generator keys such as `num_y`) |
| ADL-002 | OAS-local `BsplineComp` replaces OM3 `SplineComp(method="bsplines")` (N-001) | om4 has no SplineComp |
| ADL-003 | Approximated partials (N-002): components that were fully CS/FD use component-level `differentiator="cs"`; mixed components get analytic partials | om4 has no per-subjac approximation |
| ADL-005 | `VarDecl`/`PartialsBuffer` porting aids rather than rewriting every declaration and subjac fill by hand | Keeps the OM3 math and sparsity code verbatim, which is what makes the diff reviewable and the cross-checks meaningful. Neither is a run-time OM3 emulation: components still have typed fields and no `options`. Rejected: an `options`/`add_input` shim on the component itself (the user ruled this out) |
| ADL-006 | `VortexMesh.alpha` takes deg (TD-101) | om4 B-008 gives wrong totals for mixed-unit promoted inputs |
| ADL-004 | Spline outputs are 1-D `(n,)` rather than OM3's `(vec_size=1, n)` | Every OAS consumer declares 1-D inputs. OM3 tolerated the shape mismatch; om4 does not |

## Deferred (not in first pass)

Multi-section surfaces, `mphys/`, FFD/pygeo (`DVGeo` is a live non-serializable
object), VSP import, `plot_wing`/`plot_wingbox` (need optimizer recording), and
parallel multipoint.
