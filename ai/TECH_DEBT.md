# Technical debt — OAS om4 port

Debt taken on deliberately during the port, and OAS features not yet ported. om4-side
gaps and bugs live in `ai/OM4_NEEDS.md`. Resolved entries are struck through and kept.

Severity: **High** = wrong results possible; **Medium** = will bite in a foreseeable next
step; **Low** = tidiness.

## Not yet ported

| ID | Item | Severity | Notes |
|---|---|---|---|
| TD-001 | Multi-section surfaces (`MultiSecGeometry`, `MultiSecAerostructGeometry`, `geometry_unification`, `geometry_multi_join`, `section_mesh_generator` beyond `gen-mesh`) | Medium | `Surface` raises `NotImplementedError` on `is_multi_section`. Tests `test_multi_*` are not yet ported. OM3 code is on `main` |
| TD-002 | Compressible VLM (`CompressibleVLMStates`, `pg_scale.py`, `pg_wind_rotation.py`) | Medium | `AeroPoint(compressible=True)` raises. `pg_*` also need analytic partials for the Mach/alpha/beta columns OM3 complex-stepped (OM4_NEEDS N-002) |
| TD-003 | FFD geometry (`ffd_component.py`, `DVGeo` option) | Low | `DVGeo` is a live pygeo object and can't be a serializable field. Needs a design for referencing external geometry engines |
| TD-004 | `mphys/` | Low | mphys is built on OM3 |
| TD-005 | `plot_wing` / `plot_wingbox` | Low | Need optimizer-iteration recording, which om4 lacks |
| TD-006 | Structures, transfer, aerostructural groups, remaining functionals | — | Next phases of `ai/OM4_MIGRATION_PLAN.md` |

## Deliberate deviations from OM3

| ID | Deviation | Severity | Why / follow-up |
|---|---|---|---|
| TD-101 | `VortexMesh` input `alpha` is in **deg** (OM3: rad) | Low | Works around OM4_NEEDS B-008 (wrong totals across mixed-unit promoted inputs). The promoted `alpha` is unchanged for users, but anyone connecting directly to `aero_states.alpha` with ground effect sees the unit change. Revert once B-008 is fixed upstream |
| TD-102 | `Geometry` passes control-point values into `BsplineComp(cp_val=...)` as well as `indep_defaults` | Low | Works around OM4_NEEDS B-005 (`IndepDefault.val` ignored). Harmless once fixed |
| TD-103 | Spline outputs are 1-D `(n,)` rather than OM3's `(1, n)` | Low | ADL-004. Scripts that index `twist[0, :]` need updating |
| TD-104 | Components no longer carry `set_check_partial_options` | Low | OM4_NEEDS N-004. Per-component FD steps and CS choices that OM3 encoded must be passed explicitly by the partials tests |
| TD-105 | `VLMMtxRHSComp` and `MomentCoefficient` recompute in `compute_partials` what OM3 cached from `compute` | Low | Removes a call-order dependence. Costs one extra assembly per linearization; measure if it shows up in profiles |
