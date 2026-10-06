"""
Typed surface definition for OpenAeroStruct.

OAS on OpenMDAO 3 passed an untyped ``surface`` dict to every component and group.
om4 options are strict, serializable Pydantic fields, so the dict becomes the
``Surface`` model below (ADL-001 in ``ai/ARCHITECTURE.md``).

Legacy dicts are still accepted anywhere a ``Surface`` is expected: Pydantic
validates them into a ``Surface``.  To keep the leaf-component math identical to
the OM3 code, ``Surface`` also supports the read-only mapping protocol the old
code relied on:

- ``surface["mesh"]`` returns the field value
- ``"twist_cp" in surface`` is True only when the field was given (is not None),
  which is what ``"twist_cp" in surface.keys()`` meant for the dict
- ``surface.get("twist_cp_dv", True)`` returns the default when the field is None
"""

import warnings
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from om4.utils.types import FloatOrNpArray


_COMPOSITE_KEYS = (
    "safety_factor",
    "ply_angles",
    "ply_fractions",
    "E1",
    "E2",
    "nu12",
    "G12",
    "sigma_t1",
    "sigma_c1",
    "sigma_t2",
    "sigma_c2",
    "sigma_12max",
)


class Surface(BaseModel):
    """A single lifting surface: its geometry, discretization, and material properties."""

    model_config = ConfigDict(
        strict=True, extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )

    # --- wing definition -------------------------------------------------------------
    name: str = Field(description="Name of the surface; prefixes its variables in the model.")
    mesh: FloatOrNpArray | Literal["gen-mesh"] = Field(
        description="Mesh of shape (nx, ny, 3), or 'gen-mesh' to build it from the section keys."
    )
    symmetry: bool = Field(default=False, description="If True, model one half of the surface.")
    S_ref_type: Literal["wetted", "projected"] = Field(default="wetted")
    span: float | None = None
    taper: float | None = None
    sweep: float | None = None
    dihedral: float | None = None
    twist_cp: FloatOrNpArray | None = None
    chord_cp: FloatOrNpArray | None = None
    xshear_cp: FloatOrNpArray | None = None
    yshear_cp: FloatOrNpArray | None = None
    zshear_cp: FloatOrNpArray | None = None
    ref_axis_pos: float | None = None

    # Flags selecting whether a geometric parameter gets an input default (and so can be a
    # design variable).  None means the OM3 default of True.
    twist_cp_dv: bool | None = None
    chord_cp_dv: bool | None = None
    t_over_c_cp_dv: bool | None = None
    xshear_cp_dv: bool | None = None
    yshear_cp_dv: bool | None = None
    zshear_cp_dv: bool | None = None
    sweep_dv: bool | None = None
    span_dv: bool | None = None
    dihedral_dv: bool | None = None
    taper_dv: bool | None = None

    # --- mesh generation ('gen-mesh') --------------------------------------------------
    root_chord: float | None = None
    nx: int | None = None
    ny: int | None = None
    num_sections: int | None = None
    bpanels: Any = None
    cpanels: Any = None
    root_section: int | None = None

    # --- aerodynamics ------------------------------------------------------------------
    CL0: float = 0.0
    CD0: float = 0.0
    with_viscous: bool = False
    with_wave: bool = False
    groundplane: bool = False
    k_lam: float | None = None
    t_over_c_cp: FloatOrNpArray | None = None
    c_max_t: float | None = None

    # --- structures --------------------------------------------------------------------
    fem_model_type: Literal["tube", "wingbox"] | None = None
    E: float | None = None
    G: float | None = None
    # ``yield`` is a Python keyword, so the field is ``yield_stress`` with alias ``yield``.
    yield_stress: float | None = Field(default=None, alias="yield")
    safety_factor: float | None = None
    mrho: float | None = None
    fem_origin: float | None = None
    wing_weight_ratio: float | None = None
    exact_failure_constraint: bool | None = None
    struct_weight_relief: bool | None = None
    distributed_fuel_weight: bool | None = None
    fuel_density: float | None = None
    Wf_reserve: float | None = None
    n_point_masses: int | None = None

    # tube
    thickness_cp: FloatOrNpArray | None = None
    radius_cp: FloatOrNpArray | None = None

    # wingbox
    spar_thickness_cp: FloatOrNpArray | None = None
    skin_thickness_cp: FloatOrNpArray | None = None
    original_wingbox_airfoil_t_over_c: float | None = None
    strength_factor_for_upper_skin: float | None = None
    data_x_upper: FloatOrNpArray | None = None
    data_y_upper: FloatOrNpArray | None = None
    data_x_lower: FloatOrNpArray | None = None
    data_y_lower: FloatOrNpArray | None = None

    # composite (tsai-wu wingbox)
    useComposite: bool | None = None
    ply_angles: FloatOrNpArray | None = None
    ply_fractions: FloatOrNpArray | None = None
    E1: float | None = None
    E2: float | None = None
    nu12: float | None = None
    G12: float | None = None
    sigma_t1: float | None = None
    sigma_c1: float | None = None
    sigma_t2: float | None = None
    sigma_c2: float | None = None
    sigma_12max: float | None = None

    # FFD (deferred; see ai/TECH_DEBT.md)
    mx: int | None = None
    my: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_dict(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("is_multi_section"):
            raise NotImplementedError(
                "Multi-section surfaces are not yet ported to om4 (see ai/TECH_DEBT.md)."
            )
        data.pop("is_multi_section", None)
        # OM3-era OAS warned about and ignored unknown keys (check_surface_dict_keys), and
        # existing scripts and test fixtures rely on that (e.g. mesh-generator keys such as
        # num_x/num_y left in the surface dict), so keep that behavior rather than forbid.
        known = set(cls.model_fields) | {"yield"}
        for key in [k for k in data if k not in known]:
            warnings.warn(
                f"Key `{key}` in surface dict is (likely) not supported in OAS and will be ignored",
                category=RuntimeWarning,
                stacklevel=2,
            )
            data.pop(key)
        # Legacy dicts commonly hold numpy scalars and ints where floats are expected.
        for key, val in list(data.items()):
            if isinstance(val, np.generic):
                data[key] = val.item()
            field = cls.model_fields.get(key)
            if field is not None and isinstance(data[key], int) and not isinstance(data[key], bool):
                if field.annotation in (float, float | None):
                    data[key] = float(data[key])
        return data

    @model_validator(mode="after")
    def _check_composite(self):
        if not self.useComposite:
            return self
        missing = [k for k in _COMPOSITE_KEYS if self[k] is None]
        if missing:
            raise ValueError(
                f"{missing} not found in surface, when `useComposite` is True, the following "
                f"keys must be present: {list(_COMPOSITE_KEYS)}"
            )
        if self.fem_model_type != "wingbox":
            raise ValueError("`fem_model_type` must be 'wingbox' when `useComposite` is True")
        if len(self.ply_angles) != len(self.ply_fractions):
            raise ValueError("Length of `ply_angles` and `ply_fractions` arrays must be equal")
        frac_sum = float(np.sum(self.ply_fractions))
        if abs(frac_sum - 1) > 1e-2:
            raise ValueError(
                f"Sum of `ply_fractions` ({self.ply_fractions}) is {frac_sum} must be 1."
            )
        return self

    # --- read-only mapping protocol, for the OM3-era component math ------------------

    @staticmethod
    def _field_name(key: str) -> str:
        return "yield_stress" if key == "yield" else key

    def __getitem__(self, key: str) -> Any:
        name = self._field_name(key)
        if name not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, name)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        name = self._field_name(key)
        return name in type(self).model_fields and getattr(self, name) is not None

    def get(self, key: str, default: Any = None) -> Any:
        val = self[key] if self._field_name(key) in type(self).model_fields else None
        return default if val is None else val

    def keys(self) -> list[str]:
        """Return the keys that were given, as the legacy dict would have had them."""
        names = ("yield" if n == "yield_stress" else n for n in type(self).model_fields)
        return [k for k in names if k in self]
