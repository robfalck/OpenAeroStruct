# om4 needs found while porting OpenAeroStruct

Running list of om4 components, features, and bugs that the OAS port ran into.
Each entry says what OAS does about it in the meantime. Strike through (`~~...~~`)
when resolved upstream; don't delete. Minimal reproductions live in `ai/tools/repros/`.

## Missing components / features

| ID | Need | Where OAS hit it | Interim in OAS |
|---|---|---|---|
| N-001 | `SplineComp` (at least `method="bsplines"`; OM3 also has akima, scipy methods) | `geometry/geometry_group.py`, structures groups (twist/chord/shear/thickness control points) | OAS-local `openaerostruct/utils/bspline_comp.py:BsplineComp`, a verbatim port of OM3's B-spline basis. Matches OM3 to round-off |
| N-002 | Per-subjac approximation (OM3 `declare_partials(of, wrt, method="cs"/"fd")`). `PartialsSpec` has no `method`; only a component-wide `differentiator` | ~16 components. Done so far: `aerodynamics/eval_mtx.py` (alpha), `aerodynamics/vortex_mesh.py` (alpha, height_agl, ground effect). Pending: `pg_scale.py`, `pg_wind_rotation.py`, `structures/fuel_vol.py`, … | Mixed components get hand-derived partials, each cross-checked against OM3's CS values (`ai/tools/om3_crosscheck.py`). Fully-approximated components will use `differentiator="cs"` (verified exact on a probe) |
| N-003 | `assert_check_partials` and other testing asserts | every partials unit test | `openaerostruct/utils/testing.py:assert_check_partials` |
| N-004 | Per-component check settings (OM3 `set_check_partial_options(wrt, method, step)`) | ~17 components declare FD steps or CS for checking | Dropped from components. Tests pass the differentiator/step to `run_test` |
| N-005 | Subjacs are write-only inside `compute_partials`: `partials[of, wrt]` returns a `COOSubmat` that does not support slice assignment, in-place `+=`, or reads. OM3 code commonly fills a subjac piecewise (`partials[k][:n] = ...; partials[k][m:] += ...`) | 13 OAS modules, worst `geometry_mesh_transformations.py` (33 sites), `moment_coefficient.py` (25), `aerodynamics/geometry.py` (13) | Build the full value array locally and assign it once. Port rule 7 in `ai/ARCHITECTURE.md` |

## Bugs / behavior differences

| ID | Issue | Evidence | Interim in OAS |
|---|---|---|---|
| B-001 | `check_partials` builds the FD/CS reference **only inside the declared sparsity pattern** (`compexecnode.py:436`, `subjacs = {n: sj.copy(zero=True) ...}`), so undeclared nonzeros and undeclared `(of, wrt)` pairs are invisible. OM3 compares against a dense approximation | Probe: `y = x**2 + x[0]` with only a diagonal declared; `Jref` showed zeros in column 0 below the diagonal | `utils/testing.py:assert_sparsity_complete` does a dense FD sweep for small explicit components. Ports are also cross-checked against OM3's analytic Jacobians |
| B-002 | `get_tol_violation` returns `(0.0, 0.0, False)` for empty arrays, but callers unpack `(above_tol, max_abs, max_rel)`, so the order is wrong. Harmless today, since `0.0` is falsy | `om4/utils/array_utils.py:541` | none |
| B-003 | `check_partials` double-counts duplicate `(row, col)` pairs in the **reference** Jacobian. The analytic J sums duplicates correctly (as OM3 does) and totals are correct, but `Jref` scales a duplicated entry by its multiplicity, giving false failures | Probe: pattern with `(0,0)` declared twice, values 1 and 2; `J[0,0]=3` (correct), `Jref[0,0]=6`; fwd/rev totals both 3 | OAS keeps its duplicate-containing patterns (`Stretch`, `Sweep`, `Dihedral`, …) and checks those against dense FD via `assert_sparsity_complete` rather than om4 `check_partials` |
| B-004 | **Silent unit error.** An unconnected promoted input whose leaves declare different units, set with `Problem.set_val(name, v)` and no `InputDefault`, gets the raw number `v` in every leaf with no conversion and no error (5 → 5 deg and 5 rad). OM3 rejects this at setup and asks for `set_input_defaults(units=...)` | `ai/tools/repros/promoted_input_units_and_defaults.py`, case A. OAS hits it with ground effect: `alpha` is rad in `VortexMesh`, deg elsewhere | Every OAS group gives mixed-unit promoted inputs an `InputDefault` with units (case C converts correctly) |
| B-005 | **`InputDefault.val` is ignored.** With `input_defaults={"a": InputDefault(val=5.0, units="deg")}` and no `set_val`, each leaf keeps its own component default (0 and 1), and `get_val("a")` returns a leaf value rather than 5. The `units` part does work | Same repro, case B | Components are also constructed with the intended default (e.g. `BsplineComp(cp_val=surface["twist_cp"])`), so the leaf default agrees with the `InputDefault` |
| B-006 | Ambiguous promoted-input defaults aren't diagnosed. Leaves promoted to one name with different default values (OAS: `alpha` 0 deg in `ConvertVelocity`, 1 deg in `EvalVelMtx`) are accepted silently, and the promoted value is taken from one of them. OM3 raises and asks for `set_input_defaults` | Probe on `VLMStates`: `get_val("alpha")` = 1, `convert_velocity.alpha` = 0 before any `set_val` | Groups give these inputs explicit `InputDefault`s |

## Docs / housekeeping

| ID | Issue |
|---|---|
| H-001 | `FEATURES.md` lists pyOptSparse as 🔴, but `om4/optimizers/pyoptsparse_optimizer.py` exists and `PyOptSparseOptimizer` is exported from `om4.api`. Also lists `BalanceComp`/`InputResidsComp` as missing in the gap summary while the tables say 🟢 |
| H-002 | The OAS pixi env installs om4 editable from whatever branch `../om4.git` has checked out (currently `poem068_restart_from_successful`). Port results depend on that branch |
| H-003 | OAS's `ruff.toml` extends `~/.config/ruff/ruff.toml` (the MDO Lab shared config), which isn't present on this machine, so `ruff` fails until it's installed or the extend is dropped. (OAS housekeeping, not an om4 issue) |
