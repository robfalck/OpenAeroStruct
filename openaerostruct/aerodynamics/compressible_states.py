"""
Class definition for CompressibleVLMStates.
"""

from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.aerodynamics.get_vectors import GetVectors
from openaerostruct.aerodynamics.collocation_points import CollocationPoints
from openaerostruct.aerodynamics.eval_mtx import EvalVelMtx
from openaerostruct.aerodynamics.convert_velocity import ConvertVelocity
from openaerostruct.aerodynamics.mtx_rhs import VLMMtxRHSComp
from openaerostruct.aerodynamics.solve_matrix import SolveMatrix
from openaerostruct.aerodynamics.horseshoe_circulations import HorseshoeCirculations
from openaerostruct.aerodynamics.eval_velocities import EvalVelocities
from openaerostruct.aerodynamics.mesh_point_forces import MeshPointForces
from openaerostruct.aerodynamics.panel_forces import PanelForces
from openaerostruct.aerodynamics.panel_forces_surf import PanelForcesSurf
from openaerostruct.aerodynamics.rotational_velocity import RotationalVelocity
from openaerostruct.aerodynamics.vortex_mesh import VortexMesh
from openaerostruct.aerodynamics.pg_transform import PGTransform, InversePGTransform
from openaerostruct.utils.om4_utils import field_values, vlm_system_size
from openaerostruct.utils.surface import Surface


def _compressible_vlm_states_kwargs(surfaces: list[Surface], rotational: bool) -> dict:
    """Return the Group spec for CompressibleVLMStates: PG transform, incompressible VLM solve
    in the PG domain, then the inverse PG transform back to physical forces."""
    num_collocation_points = vlm_system_size(surfaces)
    num_force_points = num_collocation_points

    subs = {}
    connections = []

    def conn(src, tgt):
        connections.append(om.Connection(src=src, tgt=tgt))

    # ----------------------------------------------------------------
    # Step 0: Need to calculate a few things prior to transformation.
    # ----------------------------------------------------------------

    subs["collocation_points"] = CollocationPoints(surfaces=surfaces, promotes_inputs=["*"])

    if rotational:
        subs["rotational_velocity"] = RotationalVelocity(surfaces=surfaces, promotes_inputs=["cg", "omega"])
        conn("collocation_points.coll_pts", "rotational_velocity.coll_pts")

    # ----------------------------------------
    # Step 1: Transform geometry to PG domain
    # ----------------------------------------

    conn("collocation_points.coll_pts", "pg_transform.coll_pts")
    conn("collocation_points.bound_vecs", "pg_transform.bound_vecs")
    conn("collocation_points.force_pts", "pg_transform.force_pts")
    if rotational:
        conn("rotational_velocity.rotational_velocities", "pg_transform.rotational_velocities")

    prom_in = ["alpha", "beta", "Mach_number"]
    for surface in surfaces:
        name = surface["name"]

        vname = f"{name}_def_mesh"
        prom_in.append(vname)
        conn(f"pg_transform.{vname}_pg", f"vortex_mesh.{vname}")

        vname = f"{name}_normals"
        prom_in.append(vname)
        conn(f"pg_transform.{vname}_pg", f"mtx_rhs.{vname}")

    subs["pg_transform"] = PGTransform(surfaces=surfaces, rotational=rotational, promotes_inputs=prom_in)

    conn("pg_transform.coll_pts_pg", "coll_pts")
    conn("pg_transform.bound_vecs_pg", "bound_vecs")
    conn("pg_transform.force_pts_pg", "force_pts")
    if rotational:
        conn("pg_transform.rotational_velocities_pg", "rotational_velocities")

    # ---------------------------------------------------
    # Step 2: Solve incompressible problem in PG domain
    # ---------------------------------------------------

    subs["vortex_mesh"] = VortexMesh(surfaces=surfaces, promotes_outputs=["*"])

    subs["get_vectors"] = GetVectors(
        surfaces=surfaces,
        num_eval_points=num_collocation_points,
        eval_name="coll_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # In the PG domain, alpha and beta are zero.
    subs["pg_frame"] = om.ExecComp(
        exprs=["alpha_pg = 0.0", "beta_pg = 0.0"],
        outputs={"alpha_pg": {"units": "deg", "val": 0.0}, "beta_pg": {"units": "deg", "val": 0.0}},
    )

    subs["mtx_assy"] = EvalVelMtx(
        surfaces=surfaces,
        num_eval_points=num_collocation_points,
        eval_name="coll_pts",
        promotes_inputs=["*_vectors"],
        promotes_outputs=["*"],
    )
    conn("pg_frame.alpha_pg", "mtx_assy.alpha")

    # Convert freestream velocity to array of velocities.
    # Note, don't want to promote alpha/beta here because we are in the transformed system.
    prom_in_cv = ["v"]
    if rotational:
        prom_in_cv.append("rotational_velocities")
    subs["convert_velocity"] = ConvertVelocity(
        surfaces=surfaces, rotational=rotational, promotes_inputs=prom_in_cv, promotes_outputs=["*"]
    )
    conn("pg_frame.alpha_pg", "convert_velocity.alpha")
    conn("pg_frame.beta_pg", "convert_velocity.beta")

    subs["mtx_rhs"] = VLMMtxRHSComp(
        surfaces=surfaces,
        promotes_inputs=["freestream_velocities", "*coll_pts_vel_mtx"],
        promotes_outputs=["*"],
    )

    subs["solve_matrix"] = SolveMatrix(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    subs["horseshoe_circulations"] = HorseshoeCirculations(
        surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]
    )

    subs["get_vectors_force"] = GetVectors(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Note, don't want to promote alpha here because we are in the transformed system.
    subs["mtx_assy_forces"] = EvalVelMtx(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*_force_pts_vectors"],
        promotes_outputs=["*"],
    )
    conn("pg_frame.alpha_pg", "mtx_assy_forces.alpha")

    subs["eval_velocities"] = EvalVelocities(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    subs["panel_forces"] = PanelForces(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    subs["panel_forces_surf"] = PanelForcesSurf(surfaces=surfaces, promotes_inputs=["*"])

    # -----------------------------------------------------------
    # Step 3: Transform forces from PG domain to physical domain
    # -----------------------------------------------------------

    prom_out = []
    for surface in surfaces:
        vname = f"{surface['name']}_sec_forces"
        prom_out.append(vname)
        conn(f"panel_forces_surf.{vname}", f"inverse_pg_transform.{vname}_pg")

    subs["inverse_pg_transform"] = InversePGTransform(
        surfaces=surfaces, promotes_inputs=["alpha", "beta", "Mach_number"], promotes_outputs=prom_out
    )

    # ---------------------------------------------------------------
    # Step 4: Mesh point forces are downstream, already transformed.
    # ---------------------------------------------------------------

    subs["mesh_point_forces_surf"] = MeshPointForces(
        surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]
    )

    return {"subsystems": subs, "connections": connections}


class CompressibleVLMStates(om.Group):
    """
    Group that contains the states for a compressible aerodynamic analysis.

    This is done in three steps:

        1. Convert all VLM geometries to Prandtl-Glaert domain.

        2. Solve the VLM problem in the Prandtl-Glaert domain using
        VLMStates group.

        3. Convert the resulting forces back to the physical domain through
        the inverse Prandtl-Glauert transform to recover the compressible
        forces.
    """

    surfaces: list[Surface]
    rotational: bool = Field(default=False, description="Turn on support for computing angular velocities.")

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)
    connections: list[om.Connection] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_compressible_vlm_states_kwargs(f.surfaces, f.rotational))
        return data
