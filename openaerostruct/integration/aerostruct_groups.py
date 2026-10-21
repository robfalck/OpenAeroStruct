from typing import Any

import numpy as np
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.functionals.total_performance import TotalPerformance
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.structures.spatial_beam_functionals import SpatialBeamFunctionals
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.spatial_beam_states import SpatialBeamStates
from openaerostruct.structures.tube_group import TubeGroup
from openaerostruct.structures.wingbox_group import WingboxGroup
from openaerostruct.transfer.displacement_transfer_group import DisplacementTransferGroup
from openaerostruct.transfer.load_transfer import LoadTransfer
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _aerostruct_geometry_kwargs(surface: Surface, connect_geom_DVs: bool) -> dict:
    geom_in = []
    geom_out = ["mesh"]

    if connect_geom_DVs:
        # If connect_geom_DVs is true, then we promote all of the geometric design variables.
        # If it's false, then we do not promote them, which means that the geometry at each AeroStruct
        # point is independent, and the user can provide different values at each point. This is useful
        # for morphing DVs, such as twist or span, that differ between points in a multipoint scheme.
        if "twist_cp" in surface:
            geom_in.append("twist_cp")
        if "t_over_c_cp" in surface:
            geom_out.append("t_over_c")
        if "sweep" in surface:
            geom_in.append("sweep")
        if "taper" in surface:
            geom_in.append("taper")

    subs = {"geometry": Geometry(surface=surface, promotes_inputs=geom_in, promotes_outputs=geom_out)}

    fem_model_type = surface["fem_model_type"].lower()
    if fem_model_type == "tube":
        tube_in = []
        if "thickness_cp" in surface and connect_geom_DVs:
            tube_in.append("thickness_cp")
        if "radius_cp" not in surface:
            tube_in += ["mesh", "t_over_c"]
        subs["tube_group"] = TubeGroup(
            surface=surface,
            promotes_inputs=tube_in,
            promotes_outputs=["A", "Iy", "Iz", "J", "radius", "thickness"],
        )
    elif fem_model_type == "wingbox":
        # Connections and nomenclature are the same for the isotropic and composite wingbox
        wingbox_in = ["mesh", "t_over_c"]
        wingbox_out = ["A", "Iy", "Iz", "J", "Qz", "A_enc", "A_int", "htop", "hbottom", "hfront", "hrear"]
        has_skin = "skin_thickness_cp" in surface
        has_spar = "spar_thickness_cp" in surface
        if has_skin and has_spar:
            wingbox_in += ["skin_thickness_cp", "spar_thickness_cp"]
            wingbox_out += ["skin_thickness", "spar_thickness"]
        elif has_skin or has_spar:
            raise NameError("Please have both skin and spar thickness as design variables, not one or the other.")
        subs["wingbox_group"] = WingboxGroup(surface=surface, promotes_inputs=wingbox_in, promotes_outputs=wingbox_out)
    else:
        raise NameError("Please select a valid `fem_model_type` from either `tube` or `wingbox`.")

    setup_in = ["mesh", "A", "Iy", "Iz", "J"]
    if fem_model_type == "wingbox":
        setup_in.append("A_int")
    subs["struct_setup"] = SpatialBeamSetup(
        surface=surface,
        promotes_inputs=setup_in,
        promotes_outputs=["nodes", "local_stiff_transformed", "structural_mass", "cg_location", "element_mass"],
    )
    return {"subsystems": subs}


class AerostructGeometry(om.Group):
    """Geometry, structural properties and FEM setup of one aerostructural surface."""

    surface: Surface
    connect_geom_DVs: bool = True

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("DVGeo") is not None:
            raise NotImplementedError("FFD geometry (DVGeo) is not yet ported to om4; see ai/TECH_DEBT.md.")
        data.pop("DVGeo", None)
        f = field_values(cls, data)
        data.update(_aerostruct_geometry_kwargs(f.surface, f.connect_geom_DVs))
        return data


def _coupled_as_kwargs(surface: Surface) -> dict:
    promotes = []
    if surface["struct_weight_relief"]:
        promotes += ["nodes", "element_mass", "load_factor"]
    if surface["distributed_fuel_weight"]:
        promotes += ["nodes", "load_factor"]
    if "n_point_masses" in surface:
        promotes += ["point_mass_locations", "point_masses", "nodes", "load_factor", "engine_thrusts"]

    subs = {
        "struct_states": SpatialBeamStates(
            surface=surface,
            promotes_inputs=["local_stiff_transformed", "forces", "loads"] + list(dict.fromkeys(promotes)),
            promotes_outputs=["disp"],
        ),
        "def_mesh": DisplacementTransferGroup(
            surface=surface,
            promotes_inputs=["nodes", "mesh", "disp"],
            promotes_outputs=["def_mesh"],
        ),
        "aero_geom": VLMGeometry(
            surface=surface,
            promotes_inputs=["def_mesh"],
            promotes_outputs=["b_pts", "widths", "lengths_spanwise", "lengths", "chords", "normals", "S_ref"],
        ),
    }
    return {"subsystems": subs, "linear_solver": om.LinearRunOnce()}


class CoupledAS(om.Group):
    """Structural states, displacement transfer and deformed-mesh aero geometry of one surface."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_coupled_as_kwargs(f.surface))
        return data


def _coupled_performance_kwargs(surface: Surface) -> dict:
    subs = {
        "aero_funcs": VLMFunctionals(
            surface=surface,
            promotes_inputs=[
                "v",
                "alpha",
                "beta",
                "Mach_number",
                "re",
                "rho",
                "widths",
                "lengths_spanwise",
                "lengths",
                "S_ref",
                "sec_forces",
                "t_over_c",
            ],
            promotes_outputs=["CDv", "CDw", "L", "D", "CL1", "CDi", "CD", "CL", "Cl"],
        )
    }

    fem_model_type = surface["fem_model_type"].lower()
    if fem_model_type == "tube":
        subs["struct_funcs"] = SpatialBeamFunctionals(
            surface=surface,
            promotes_inputs=["thickness", "radius", "nodes", "disp"],
            promotes_outputs=["thickness_intersects", "vonmises", "failure"],
        )
    elif fem_model_type == "wingbox":
        stress_name = "tsaiwu_sr" if surface.get("useComposite", False) else "vonmises"
        subs["struct_funcs"] = SpatialBeamFunctionals(
            surface=surface,
            promotes_inputs=[
                "Qz",
                "J",
                "A_enc",
                "spar_thickness",
                "htop",
                "hbottom",
                "hfront",
                "hrear",
                "nodes",
                "disp",
            ],
            promotes_outputs=[stress_name, "failure"],
        )
    else:
        raise NameError("Please select a valid `fem_model_type` from either `tube` or `wingbox`.")
    return {"subsystems": subs}


class CoupledPerformance(om.Group):
    """Aerodynamic and structural functionals of one surface, evaluated after the coupled solve."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_coupled_performance_kwargs(f.surface))
        return data


def _aerostruct_point_kwargs(
    surfaces: list[Surface],
    user_specified_Sref: bool,
    internally_connect_fuelburn: bool,
    compressible: bool,
    rotational: bool,
    fen: bool,
) -> dict:
    if compressible:
        raise NotImplementedError("Compressible VLM states are not yet ported to om4; see ai/TECH_DEBT.md.")

    coupled_subs = {}
    coupled_conns = []
    connections = []

    def cconn(src, tgt):
        coupled_conns.append(om.Connection(src=src, tgt=tgt))

    def conn(src, tgt):
        connections.append(om.Connection(src=src, tgt=tgt))

    for surface in surfaces:
        name = surface["name"]

        # Connect the output of the loads component with the FEM displacement parameter.
        # This links the coupling within the coupled group that necessitates the subgroup solver.
        cconn(name + "_loads.loads", name + ".loads")

        # Perform the connections with the modified names within the 'aero_states' group.
        cconn(name + ".normals", "aero_states." + name + "_normals")
        cconn(name + ".def_mesh", "aero_states." + name + "_def_mesh")

        # Connect the results from 'coupled' to the performance groups
        cconn(name + ".def_mesh", name + "_loads.def_mesh")
        cconn("aero_states." + name + "_sec_forces", name + "_loads.sec_forces")

        conn("coupled.aero_states." + name + "_sec_forces", name + "_perf.sec_forces")

        # Connection performance functional variables
        conn(name + "_perf.CL", "total_perf." + name + "_CL")
        conn(name + "_perf.CD", "total_perf." + name + "_CD")
        conn("coupled.aero_states." + name + "_sec_forces", "total_perf." + name + "_sec_forces")
        conn("coupled." + name + ".chords", name + "_perf.aero_funcs.chords")

        # Connect parameters from the 'coupled' group to the performance groups for the individual surfaces.
        conn("coupled." + name + ".disp", name + "_perf.disp")
        conn("coupled." + name + ".S_ref", name + "_perf.S_ref")
        conn("coupled." + name + ".widths", name + "_perf.widths")
        conn("coupled." + name + ".lengths", name + "_perf.lengths")
        conn("coupled." + name + ".lengths_spanwise", name + "_perf.lengths_spanwise")

        # Connect parameters from the 'coupled' group to the total performance group.
        conn("coupled." + name + ".S_ref", "total_perf." + name + "_S_ref")
        conn("coupled." + name + ".widths", "total_perf." + name + "_widths")
        conn("coupled." + name + ".chords", "total_perf." + name + "_chords")
        conn("coupled." + name + ".b_pts", "total_perf." + name + "_b_pts")

        # The 'coupled' group must contain all components and parameters needed to converge the
        # aerostructural system.
        if surface["distributed_fuel_weight"] or "n_point_masses" in surface or surface["struct_weight_relief"]:
            prom_in = ["load_factor"]
        else:
            prom_in = []
        coupled_subs[name] = CoupledAS(surface=surface, promotes_inputs=prom_in)

    ground_effect = any(surface.get("groundplane", False) for surface in surfaces)

    prom_in = ["v", "alpha", "beta", "rho"]
    if ground_effect:
        prom_in.append("height_agl")
    if rotational:
        prom_in.extend(["omega", ("cg", "empty_cg")])

    # A single 'aero_states' group for the whole system within the coupled group.
    coupled_subs["aero_states"] = VLMStates(surfaces=surfaces, rotational=rotational, promotes_inputs=prom_in)

    for surface in surfaces:
        coupled_subs[surface["name"] + "_loads"] = LoadTransfer(surface=surface)

    coupled_prom_in = ["v", "alpha", "beta", "rho"]
    if ground_effect:
        coupled_prom_in.append("height_agl")
    if rotational:
        coupled_prom_in.extend(["omega", "empty_cg"])

    input_defaults = {
        "beta": om.InputDefault(val=np.zeros(1), units="deg"),
        "alpha": om.InputDefault(val=np.zeros(1), units="deg"),
        "v": om.InputDefault(val=np.ones(1), units="m/s"),
        "rho": om.InputDefault(val=np.ones(1), units="kg/m**3"),
        "Mach_number": om.InputDefault(val=np.zeros(1)),
    }

    interface_inputs = None
    interface_outputs = None
    coupled_input_defaults = {}
    if fen:
        # The coupled group's inputs that are never satisfied by an internal connection or
        # same-name promotion: the globally promoted ones plus, per surface, the CoupledAS
        # boundary inputs that every caller wires in from AerostructGeometry.
        interface_inputs = list(coupled_prom_in)
        for surface in surfaces:
            name = surface["name"]
            interface_inputs += [f"{name}.local_stiff_transformed", f"{name}.nodes", f"{name}.mesh"]

        # Every output of `coupled` that a sibling (performance group) consumes: exactly the
        # sources already recorded in `connections` that live inside `coupled`.
        seen = set()
        interface_outputs = []
        for c in connections:
            if c.src.startswith("coupled."):
                rel = c.src[len("coupled.") :]
                if rel not in seen:
                    seen.add(rel)
                    interface_outputs.append(rel)

        # A FEN is its own connection-graph domain, so promoted-name default ambiguities
        # (e.g. `alpha` at 0 deg vs 1 deg across its leaves) are no longer resolved by
        # AerostructPoint's own `input_defaults` below; `coupled` needs its own copy for
        # whichever ambiguous names it actually promotes.
        coupled_input_defaults = {k: v for k, v in input_defaults.items() if k in coupled_prom_in}

    coupled = om.Group(
        subsystems=coupled_subs,
        connections=coupled_conns,
        promotes_inputs=coupled_prom_in,
        nonlinear_solver=om.NonlinearBlockGS(
            relaxation=om.AitkenRelaxation(),
            maxiter=100,
            atol=1e-7,
            rtol=1e-30,
            iprint=2,
            err_on_non_converge=True,
        ),
        linear_solver=om.DirectSolver(assembled_format="sparse"),
        interface_inputs=interface_inputs,
        interface_outputs=interface_outputs,
        input_defaults=coupled_input_defaults,
    )

    subs = {"coupled": coupled}
    for surface in surfaces:
        subs[surface["name"] + "_perf"] = CoupledPerformance(
            surface=surface, promotes_inputs=["rho", "v", "alpha", "beta", "re", "Mach_number"]
        )

    # Only the interesting results are promoted here; not all of the parameters.
    subs["total_perf"] = TotalPerformance(
        surfaces=surfaces,
        user_specified_Sref=user_specified_Sref,
        internally_connect_fuelburn=internally_connect_fuelburn,
        promotes_inputs=[
            "v",
            "rho",
            "empty_cg",
            "total_weight",
            "CT",
            "speed_of_sound",
            "R",
            "Mach_number",
            "W0",
            "load_factor",
            "S_ref_total",
        ],
        promotes_outputs=["L_equals_W", "fuelburn", "CL", "CD", "CM", "cg"],
    )

    return {"subsystems": subs, "connections": connections, "input_defaults": input_defaults}


class AerostructPoint(om.Group):
    """Single-point aerostructural analysis: the coupled aero-structural solve plus the performance functionals."""

    surfaces: list[Surface]
    user_specified_Sref: bool = False
    internally_connect_fuelburn: bool = True
    compressible: bool = Field(
        default=False, description="Compressibility correction for moderate Mach number flows (not yet ported)."
    )
    rotational: bool = Field(default=False, description="Turn on support for computing angular velocities.")
    fen: bool = Field(
        default=False,
        description="Wrap the coupled group as a FunctionalExecutionNode, hiding its internal states "
        "from the outer linear system and exposing only its interface inputs/outputs.",
    )

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[om.Connection] = Field(default=PydanticUndefined, init=False)
    input_defaults: dict[str, om.InputDefault] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(
            _aerostruct_point_kwargs(
                f.surfaces,
                f.user_specified_Sref,
                f.internally_connect_fuelburn,
                f.compressible,
                f.rotational,
                f.fen,
            )
        )
        return data
