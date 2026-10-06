from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _aero_point_kwargs(surfaces: list[Surface], user_specified_Sref: bool, rotational: bool, compressible: bool) -> dict:
    """Return the Group spec for AeroPoint: per-surface geometry, the VLM states, and performance."""
    if compressible:
        raise NotImplementedError("Compressible VLM states are not yet ported to om4; see ai/TECH_DEBT.md.")

    subs = {}
    connections = []

    def conn(src, tgt):
        connections.append(om.Connection(src=src, tgt=tgt))

    # Loop through each surface and connect relevant parameters
    for surface in surfaces:
        name = surface["name"]

        conn(name + ".normals", "aero_states." + name + "_normals")

        # Connect the results from 'aero_states' to the performance groups
        conn("aero_states." + name + "_sec_forces", name + "_perf" + ".sec_forces")

        # Connect S_ref for performance calcs
        conn(name + ".S_ref", name + "_perf.S_ref")
        conn(name + ".widths", name + "_perf.widths")
        conn(name + ".chords", name + "_perf.chords")
        conn(name + ".lengths", name + "_perf.lengths")
        conn(name + ".lengths_spanwise", name + "_perf.lengths_spanwise")

        # Connect S_ref for performance calcs
        conn(name + ".S_ref", "total_perf." + name + "_S_ref")
        conn(name + ".widths", "total_perf." + name + "_widths")
        conn(name + ".chords", "total_perf." + name + "_chords")
        conn(name + ".b_pts", "total_perf." + name + "_b_pts")
        conn(name + "_perf" + ".CL", "total_perf." + name + "_CL")
        conn(name + "_perf" + ".CD", "total_perf." + name + "_CD")
        conn("aero_states." + name + "_sec_forces", "total_perf." + name + "_sec_forces")

        subs[name] = VLMGeometry(surface=surface)

    # Add a single 'aero_states' group that solves for the circulations and forces
    # from all the surfaces, since each surface interacts with the others.
    ground_effect = any(surface.get("groundplane", False) for surface in surfaces)

    prom_in = ["v", "alpha", "beta", "rho"]
    if ground_effect:
        prom_in.append("height_agl")
    if rotational:
        prom_in.extend(["omega", "cg"])

    subs["aero_states"] = om.Subsystem(
        VLMStates(surfaces=surfaces, rotational=rotational, linear_solver=om.LinearRunOnce()),
        promotes_inputs=prom_in,
        promotes_outputs=["circulations"],
    )

    for surface in surfaces:
        subs[surface["name"] + "_perf"] = om.Subsystem(
            VLMFunctionals(surface=surface),
            promotes_inputs=["v", "alpha", "beta", "Mach_number", "re", "rho"],
        )

    # Total aero performance (CL, CD, CM) of the aircraft, accounting for all lifting surfaces.
    subs["total_perf"] = om.Subsystem(
        TotalAeroPerformance(surfaces=surfaces, user_specified_Sref=user_specified_Sref),
        promotes_inputs=["v", "rho", "cg", "S_ref_total"],
        promotes_outputs=["CM", "CL", "CD"],
    )

    indep_defaults = {
        # beta is often unused (unconnected), so give it a default value and units
        "beta": om.IndepDefault(val=np.zeros(1), units="deg"),
        # alpha's leaves disagree on their default value (0 vs 1 deg), which om4 rejects
        # at setup unless the promoted input gets an IndepDefault
        "alpha": om.IndepDefault(val=np.zeros(1), units="deg"),
        # MomentCoefficient's defaults (10 m/s, 3 kg/m**3) differ from every other leaf's 1.0
        "v": om.IndepDefault(val=np.ones(1), units="m/s"),
        "rho": om.IndepDefault(val=np.ones(1), units="kg/m**3"),
    }

    return {"subsystems": subs, "connections": connections, "indep_defaults": indep_defaults}


class AeroPoint(om.Group):
    """
    This group contains all the components needed for a single-point aerodynamic
    analysis. You would have one instance of `AeroPoint` for each flight
    condition you want to study.
    """

    surfaces: list[Surface]
    user_specified_Sref: bool = False
    rotational: bool = Field(default=False, description="Turn on support for computing angular velocities.")
    compressible: bool = Field(
        default=False, description="Compressibility correction for moderate Mach number flows (not yet ported)."
    )

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, om.Subsystem | Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[om.Connection] = Field(default=PydanticUndefined, init=False)
    indep_defaults: dict[str, om.IndepDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_aero_point_kwargs(f.surfaces, f.user_specified_Sref, f.rotational, f.compressible))
        return data
