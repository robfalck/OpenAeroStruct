# Technical debt — OAS om4 port

Debt taken on deliberately during the port, and OAS features not yet ported. om4-side
gaps and bugs live in `ai/OM4_NEEDS.md`. Resolved entries are struck through and kept.

Severity: **High** = wrong results possible; **Medium** = will bite in a foreseeable next
step; **Low** = tidiness.

## Not yet ported

| ID | Item | Severity | Notes |
|---|---|---|---|
| TD-001 | Multi-section surfaces (`MultiSecGeometry`, `MultiSecAerostructGeometry` (not ported; the OM3 version is in git history of `integration/aerostruct_groups.py` on `main`), `section_mesh_generator` beyond `gen-mesh`; the `geometry_unification` and `geometry_multi_join` components are ported and unit-tested with section dicts) | Medium | `Surface` raises `NotImplementedError` on `is_multi_section`. Tests `test_multi_*` are not yet ported. OM3 code is on `main` |
| TD-002 | Compressible VLM (`CompressibleVLMStates`, `pg_scale.py`, `pg_wind_rotation.py`) | Medium | `AeroPoint(compressible=True)` raises. `pg_*` also need analytic partials for the Mach/alpha/beta columns OM3 complex-stepped (single differentiator per component, OM4_NEEDS N-002) |
| TD-003 | FFD geometry (`ffd_component.py`, `DVGeo` option) | Low | `DVGeo` is a live pygeo object and can't be a serializable field. Needs a design for referencing external geometry engines |
| TD-004 | `mphys/` | Low | mphys is built on OM3 |
| TD-005 | `plot_wing` / `plot_wingbox` | Low | Need optimizer-iteration recording, which om4 lacks |
| TD-006 | Aerostructural integration tests (the groups `AerostructGeometry`, `CoupledAS`, `CoupledPerformance`, `AerostructPoint` are ported; only `test_aerostruct_analysis.py` is) | Medium | Remaining `test_aerostruct*`, `test_v1_aerostruct_*`, `test_wingbox_derivs/distributed_fuel`, `test_multipoint_*`, `test_morphing_*`, `test_scaneagle`, `test_simple_rect_AS` still use OM3 APIs |

## Deliberate deviations from OM3

| ID | Deviation | Severity | Why / follow-up |
|---|---|---|---|
| TD-102 | `Geometry` passes control-point values into `BsplineComp(cp_val=...)` as well as `input_defaults` | Low | Works around OM4_NEEDS B-005 (`InputDefault.val` ignored). Harmless once fixed |
| TD-103 | Spline outputs are 1-D `(n,)` rather than OM3's `(1, n)` | Low | ADL-004. Scripts that index `twist[0, :]` need updating |
| TD-104 | `set_check_partial_options(wrt=...)` becomes a component-wide `check_partials_options` | Low | OM4_NEEDS N-004. The check differentiator is chosen per component, not per `wrt`, so a component with one FD-only input is checked with FD for all of them |
| TD-105 | `VLMMtxRHSComp` and `MomentCoefficient` recompute in `compute_partials` what OM3 cached from `compute` | Low | Removes a call-order dependence. Costs one extra assembly per linearization; measure if it shows up in profiles |
| TD-106 | Structures leaf changes beyond a mechanical port: `StructureWeightLoads.element_lengths` is computed (OM3 left it unset), `WingboxFuelVolDelta` no longer mutates its inputs, `Length` uses the complex-safe `norm`, `CreateRHS` allocates a complex-safe buffer | Low | Needed for complex-step and om4's read-only inputs; values are unchanged |
| TD-107 | `AerostructPoint` keeps OM3's wiring in which `load_factor` is promoted from each `CoupledAS` into `coupled` but not out of it, so the coupled structure sees its default `load_factor`, not the one given to `total_perf` | Medium | Reproduced for parity from reading the code, not yet verified at runtime. Check whether it is a latent OM3 bug (matters when `load_factor != 1` with weight relief, fuel or point masses) |
| TD-108 | Tolerances for `check_partials` on CS-differentiated components (vonmises, fuel_loads, wingbox) are loosened in integration tests (`atol=1e-1, rtol=1e-4`) | Low | om4 switches the reference to 4th-order central FD when the component differentiator is complex step, which is noisy for large Jacobians. Consider an OM4_NEEDS entry if this keeps recurring |
| TD-109 | `test_add_point_masses.test_simple_values` is skipped | Low | Reference values assume g=9.8, `grav_constant` is 9.80665; it was already skipped upstream |
| TD-110 | `AerostructPoint` has `input_defaults` for `alpha`, `beta`, `v`, `rho`, `Mach_number` | Low | Leaves disagree on defaults and om4 rejects ambiguous promoted inputs; the values are placeholders overridden by `set_val` |
