# OM3 cross-checks

Each script compares ported om4 components or groups with the OM3 originals from `main`
(via `ai/tools/om3_crosscheck.py`): outputs, dense analytic partials for components, and
fwd/rev total derivatives for groups. Run from the repo root in the pixi env, for example:

    pixi run python ai/tools/crosschecks/xc_aeropoint.py

| Script | Covers |
|---|---|
| `xc_gmt.py` | 9 mesh transformation components |
| `xc_gm.py`, `xc_geom.py` | `GeometryMesh`, `Geometry` groups (values and totals) |
| `xc_vlmgeom.py` | `VLMGeometry` |
| `xc_vlm1.py`, `xc_vlm2.py`, `xc_vm.py` | VLMStates components, including ground effect |
| `xc_states.py` | `VLMStates` group (values, fwd/rev totals) |
| `xc_func.py` | Viscous/wave drag, lift, coefficients, totals, moment |
| `xc_aeropoint.py` | Full `Geometry` + `AeroPoint` model in four configurations (values, fwd/rev totals) |
