# OpenAeroStruct → OpenMDAO 4 migration plan

Branch: `om4`. Target framework: `om4` (sibling checkout at `../om4.git`).

## 1. What we are dealing with

**OM4 is not a drop-in API** (`om4.git/ai/ARCHITECTURE.md`). This is a structural port, not
a search-and-replace. The precedent is `dymos4.git`, which ported dymos and wrote down its
conventions in `dymos4.git/ai/ARCHITECTURE.md`. We reuse those conventions.

| OM3 idiom used by OAS | OM4 equivalent |
|---|---|
| `initialize()` + `self.options.declare(...)` (184 sites) | Typed Pydantic fields in the class body (`strict=True`, `extra="forbid"`) |
| `setup()` with `add_input`/`add_output` | `inputs`/`outputs` dicts of `InputVar`/`OutputVar`, built in a `model_validator(mode="before")` or `model_post_init` |
| `declare_partials(...)` (201 sites) | `partials: list[PartialsSpec]` (rows/cols/val/diagonal) |
| `compute(inputs, outputs)` | `compute_outputs(inputs, outputs)` |
| `apply_nonlinear` / `linearize` (ImplicitComponent) | `compute_residuals` / `compute_partials` (`solve_nonlinear`, `solve_linear` keep their names) |
| `Group.setup()` with `add_subsystem`/`connect`/`promotes` (270 / 387 / 375 sites) | Declarative `subsystems={name: Subsystem(...)}`, `connections=[Connection(...)]`. dymos4 pattern: a `_thing_kwargs(...)` builder plus a `Group` subclass whose before-validator calls it |
| `set_input_defaults` | `indep_defaults={name: IndepDefault(...)}` |
| `om.IndepVarComp` (31 files) | Not needed: unconnected inputs are set with `Problem.set_val`, and `DesignVar` must sit on an unconnected input |
| `add_design_var`/`add_constraint`/`add_objective` | `design_vars`/`constraints`/`objectives` fields on a `System`; `ref`/`scaler` become `AffineTransform.from_ref_ref0` / `from_scaler_adder` |
| `ScipyOptimizeDriver` + `run_driver` (21 files) | `ScipyMinimizeOptimizer` on the root group's `optimizer` field + `Problem.run_optimizer()` → `OptimizerResult` |
| `pyOptSparseDriver` (4 files) | `om4/optimizers/pyoptsparse_optimizer.py` (exists, though `FEATURES.md` still lists it as missing) |
| `NonlinearBlockGS(use_aitken=True)` | `NonlinearBlockGS(relaxation=AitkenRelaxation())` |
| `DirectSolver(assemble_jac=True)` + `assembled_jac_type="csc"` | `DirectSolver(assembled_format="sparse")` |
| `LinearBlockGS`, `LinearRunOnce`, `NewtonSolver`, `ExecComp`, `BalanceComp` | Present in OM4 with renamed/typed fields |
| `declare_partials(..., method="cs"/"fd")` (~16 components) | No per-subjac approximation in `PartialsSpec`. Only component-level `differentiator="cs"/"fd"` is available. See §4 D3 |
| `om.SplineComp(method="bsplines")` (geometry group) | **Missing in OM4.** See §4 D4 |
| `prob.setup(force_alloc_complex=True)` | Removed. CS stores are local to the CS node (ADL-019) |
| `check_partials(method=..., compact_print=...)`, `set_check_partial_options`, `assert_check_partials` | `check_partials` exists with a different signature/result. No `assert_check_partials`, no per-component check options. A local helper is needed |
| `SqliteRecorder` / `CaseReader` on the driver | Solver and Problem scopes only; no optimizer recording |
| Instance state such as `self.lu`, `self.system_size` set in `setup` | Declared fields, or `PrivateAttr` for runtime caches |

What OAS does **not** use, which keeps the job smaller: matrix-free APIs, `guess_nonlinear`,
`configure`, MPI/distributed variables, and discrete variables. All of those appear only under
`mphys/` and `docs/`. The two implicit components are `SolveMatrix` and the FEM solve.

Size: about 160 library modules (≈26k lines including docs scripts), about 95 component
classes, 24 group classes, 138 test files, and 206 tests.

## 2. Guiding conventions (adopted from dymos4)

1. Port the *intent*, not the structure. No `setup`/`configure` two-pass code.
2. Leaf components keep their math verbatim. Only the framework shell changes. This keeps the
   numerical diff reviewable.
3. Every ported component comes with its partials test in the same commit, against the same
   tolerances as the OM3 test.
4. Integration tests keep their **hard-coded OM3 reference values** (CL, CD, fuelburn,
   failure, …). Those numbers are the acceptance criteria for the port. Do not loosen a
   tolerance without writing down why.
5. Record decisions in `ai/ARCHITECTURE.md` (ADL table) and debt in `ai/TECH_DEBT.md`, as
   dymos4 does.

## 3. Phases

### Phase 0: environment and scaffolding
- Add `pixi.toml`, modeled on dymos4: editable `om4` from `../om4.git`, the win-64 BLAS pin,
  and `testflo`, `ruff`, `matplotlib`. Add an `opt` feature for pyoptsparse.
- Move `setup.py` to a `pyproject.toml` that depends on `om4` instead of `openmdao`.
- Write the OAS `ai/ARCHITECTURE.md` (conventions, ADL table, open questions) and `CLAUDE.md`.
- Capture an OM3 baseline: run the current suite on `main` and save timings and pass/fail. This
  gives a performance comparison later and flags tests that already fail.

### Phase 1: foundations everything else needs
- **`Surface` Pydantic model** (§4 D1), replacing the untyped `surface` dict and absorbing
  `check_surface_dict.py`. It must accept the legacy dict via `Surface.model_validate(d)` so
  user scripts keep working.
- **B-spline component**, an OAS-local `BsplineComp` (§4 D4) with tests.
- **Test helpers** in `utils/testing.py`: `run_test` and `assert_check_partials` rewritten
  for the OM4 `check_partials`, plus per-component FD/CS step overrides to replace
  `set_check_partial_options`.
- A small `oas_component` helper or base class if the repeated mesh-shape bookkeeping
  (`nx`, `ny` derived from `surface.mesh`) proves worth sharing. Decide after porting the
  first 5 components.

### Phase 2: leaf components, bottom-up (≈95 classes)
Order follows the dependency chain, so each stage can be tested on its own:

1. `common/` (atmosphere, Reynolds) and `utils/` math helpers (`vector_algebra`,
   `interpolation`: no framework code)
2. `geometry/` (mesh transformations, radius, monotonic constraint, unification, multi-join)
3. `aerodynamics/` (23 test files; includes the implicit `SolveMatrix`)
4. `structures/` (21 test files; includes the implicit FEM solve and the components with
   `method="cs"` partials)
5. `transfer/`, then `functionals/`

Each package is one PR-sized chunk: components plus their `tests/*_tests` ports. The
`method="cs"` components get the D3 treatment here.

### Phase 3: groups
`Geometry`/`MultiSecGeometry` → `AeroPoint` → `SpatialBeamAlone`, `TubeGroup`,
`WingboxGroup` → `AerostructGeometry` → `CoupledAS`/`CoupledPerformance` →
`AerostructPoint` → multipoint composition.

Things to watch:
- OM4's `collapse_by="cycle"` builds the execution tree from actual cycles. The coupled
  aerostruct solver must sit on the group that owns the cycle. dymos4 ADL-001 also warns that
  a non-RunOnce linear solver on a node with no cycle triggers a warning.
- Port the solver configuration to typed fields: NLBGS + Aitken, `atol=1e-7`,
  `err_on_non_converge`, and sparse `DirectSolver`.
- Promotions that match nothing are errors in OM4. OAS's `promotes_inputs=[...]` lists that
  depend on surface options must be built conditionally.

### Phase 4: integration and optimization tests (68 files)
- Analysis tests first (`*_analysis*.py`): no optimizer, pure value regression.
- Then SLSQP optimizations through `ScipyMinimizeOptimizer`, and the derivative tests
  (`test_wingbox_derivs`, `test_multipoint_wingbox_derivs`) through `check_totals`.
- Then the pyoptsparse tests.

### Phase 5: examples, docs, deferred features
Port `examples/` and the docs scripts last. Then work through the deferred list in §4 D5.

## 4. Decisions to make before coding (recommendations in bold)

- **D1. Surface options.** OM4 options must be strict, serializable fields. **Recommend a
  typed `Surface` model** (numpy mesh as `FloatOrNpArray`, enums for `fem_model_type`,
  `S_ref_type`, …), passed as `surface: Surface` on components and groups, with legacy-dict
  coercion. The alternative, `surface: dict[str, Any]`, is faster to port but loses
  validation, breaks JSON round-trip, and fights `strict=True`.
- **D2. API compatibility.** **Recommend breaking the OM3-style API** (like dymos4) but keeping
  the physics, variable names, and promoted names identical, so test reference values carry
  over unchanged. Keep the package name `openaerostruct`.
- **D3. `method="cs"` partials.** OM4 has no per-subjac approximation, and `FEATURES.md` marks
  complex step 🔴/partial. Options per component: (a) component-level `differentiator="cs"`;
  (b) derive analytic partials; (c) rewrite as `JaxExplicitComp` (asdex sparsity). **Recommend
  (a) for components that are fully CS today** (`fuel_loads`, `fuel_vol`,
  `section_properties_wingbox`, `tsaiwu_wingbox`, point-mass/thrust loads), **and (b) for the
  mixed ones** (`pg_scale`, `pg_wind_rotation`, `eval_mtx` alpha, `vortex_mesh`,
  `spar_within_wing`), since the CS block is usually one scalar column (alpha, beta, Mach).
  Validate OM4's component CS early. If it is not solid, upstream fixes go to om4.
- **D4. SplineComp.** **Recommend an OAS-local `BsplineComp`** (OAS had one before OM3 grew
  `SplineComp`). The alternative is to contribute SplineComp/InterpND to om4 first, which
  helps everyone but puts this port behind that work.
- **D5. Deferred or out of scope for the first pass:** `mphys/` (mphys is built on OM3),
  `ffd_component.py`/pygeo (the `DVGeo` option is a live non-serializable object), VSP mesh
  import, `plot_wing`/`plot_wingbox` (they need optimizer-case recording, which OM4 lacks), and
  `test_multipoint_parallel`. **Recommend marking these tests `skip` with a TECH_DEBT entry**
  rather than deleting them.
- **D6. Where fixes land.** Gaps found in om4 (SplineComp, CS maturity, `assert_check_partials`,
  optimizer recording) are logged in `ai/TECH_DEBT.md` and fixed upstream in `om4.git` on
  separate branches, not worked around silently in OAS.

## 5. Risks

| Risk | Mitigation |
|---|---|
| OM4 complex step is immature; ~16 components and many partials tests depend on it | Spike it in Phase 1 on `fuel_vol` before committing to D3(a) |
| Optimizations converge to slightly different optima (new scaling/transform plumbing, SLSQP wrapper differences) | Analysis tests gate first; for optimizations compare objective at the optimum, then DVs, and log any drift |
| Performance of the large VLM/FEM dense Jacobians under OM4 assembly | Compare against the Phase 0 baseline on `test_aerostruct` and the wingbox opt |
| Win-64 BLAS crashes (seen in om4/dymos4) | Reuse dymos4's pixi pin |
| Promoted-name collisions or "matched nothing" errors from OAS's conditional promotes | Build promote lists from `Surface` fields in one place per group |

## 6. Suggested first milestone

Phase 0, the `Surface` model, the test helpers, and the complex-step spike, followed by a
single end-to-end vertical slice: `test_simple_rect_aero` (the geometry → VLM aero path, about
25 components) passing under OM4 with OM3's reference values. That exercises every pattern
above except the structures and coupling, and shows the real per-component porting cost
before the rest is scheduled.
