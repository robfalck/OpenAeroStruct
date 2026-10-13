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
from openaerostruct.aerodynamics.rotational_velocity import RotationalVelocity
from openaerostruct.aerodynamics.mesh_point_forces import MeshPointForces
from openaerostruct.aerodynamics.panel_forces import PanelForces
from openaerostruct.aerodynamics.panel_forces_surf import PanelForcesSurf
from openaerostruct.aerodynamics.vortex_mesh import VortexMesh
from openaerostruct.utils.om4_utils import field_values, vlm_system_size
from openaerostruct.utils.surface import Surface


def _vlm_states_kwargs(surfaces: list[Surface], rotational: bool) -> dict:
    """Return the Group spec for VLMStates: everything from deformed meshes to panel forces."""
    num_collocation_points = vlm_system_size(surfaces)
    num_force_points = num_collocation_points

    subs = {}

    # Get collocation points
    subs["collocation_points"] = CollocationPoints(
        surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["coll_pts", "force_pts", "bound_vecs"]
    )

    # Compute the vortex mesh based off the deformed aerodynamic mesh
    subs["vortex_mesh"] = VortexMesh(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    # Get vectors from mesh points to collocation points
    subs["get_vectors"] = GetVectors(
        surfaces=surfaces,
        num_eval_points=num_collocation_points,
        eval_name="coll_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Construct matrix based on rings, not horseshoes
    subs["mtx_assy"] = EvalVelMtx(
        surfaces=surfaces,
        num_eval_points=num_collocation_points,
        eval_name="coll_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Convert freestream velocity to array of velocities
    if rotational:
        subs["rotational_velocity"] = RotationalVelocity(
            surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]
        )

    subs["convert_velocity"] = ConvertVelocity(
        surfaces=surfaces, rotational=rotational, promotes_inputs=["*"], promotes_outputs=["*"]
    )

    # Construct RHS and full matrix of system
    subs["mtx_rhs"] = VLMMtxRHSComp(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    # Solve Mtx RHS to get ring circs
    subs["solve_matrix"] = SolveMatrix(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    # Convert ring circs to horseshoe circs
    subs["horseshoe_circulations"] = HorseshoeCirculations(
        surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]
    )

    # Eval force vectors
    subs["get_vectors_force"] = GetVectors(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Set up force mtx
    subs["mtx_assy_forces"] = EvalVelMtx(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Multiply by horseshoe circs to get velocities
    subs["eval_velocities"] = EvalVelocities(
        surfaces=surfaces,
        num_eval_points=num_force_points,
        eval_name="force_pts",
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    # Get sectional panel forces
    subs["panel_forces"] = PanelForces(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    # Get panel forces for each lifting surface individually
    subs["panel_forces_surf"] = PanelForcesSurf(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    # Get nodal forces for each lifting surface individually
    subs["mesh_point_forces_surf"] = MeshPointForces(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"])

    return {"subsystems": subs}


class VLMStates(om.Group):
    """
    Group that houses all components to compute the aerodynamic states.
    """

    surfaces: list[Surface]
    rotational: bool = Field(default=False, description="Turn on support for computing angular velocities.")

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_vlm_states_kwargs(f.surfaces, f.rotational))
        return data
