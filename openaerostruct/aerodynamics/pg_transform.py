from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.aerodynamics.pg_wind_rotation import RotateToWindFrame, RotateFromWindFrame
from openaerostruct.aerodynamics.pg_scale import ScaleToPrandtlGlauert, ScaleFromPrandtlGlauert
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _pg_transform_kwargs(surfaces: list[Surface], rotational: bool) -> dict:
    """Return the Group spec for PGTransform: rotate to wind frame, then scale to PG."""
    return {
        "subsystems": {
            "rotate": RotateToWindFrame(
                surfaces=surfaces, rotational=rotational, promotes_inputs=["*"], promotes_outputs=["*"]
            ),
            "scale": ScaleToPrandtlGlauert(
                surfaces=surfaces, rotational=rotational, promotes_inputs=["*"], promotes_outputs=["*"]
            ),
        }
    }


class PGTransform(om.Group):
    """
    This group is responsible for transforming the VLM geometries from
    physical coordinates to Prandtl-Glauert coordinates. This allows the
    compressible aerodynamic problem to be solved as an equivelent
    incompressible problem.

    The transform can be broken down into two steps:

        1. Rotate the geometry from the body frame to the wind frame so
        x-axis is parallel to freestream velocity.

        2. Scale wind frame coordinates by Prandtl-Glauert factor to retrieve
        equivalent Prandtl-Glauert geometry.
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
        data.update(_pg_transform_kwargs(f.surfaces, f.rotational))
        return data


def _inverse_pg_transform_kwargs(surfaces: list[Surface]) -> dict:
    """Return the Group spec for InversePGTransform: scale back to wind frame, then rotate to aero frame."""
    return {
        "subsystems": {
            "scale": ScaleFromPrandtlGlauert(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]),
            "rotate": RotateFromWindFrame(surfaces=surfaces, promotes_inputs=["*"], promotes_outputs=["*"]),
        }
    }


class InversePGTransform(om.Group):
    """
    This group is responsible for transforming the solved incompressible
    forces in the Prandtl-Glauert domain to the compressible forces in the
    physical aerodynamic frame. This is the reverse procedure used in the
    PGTransform group.

    The inverse transform can be broken down into two steps:

        1. Scale Prandtl-Glauert force vectors to physical force vectors in wind
        frame.

        2. Rotate physical forces from wind frame to the aerodynamic frame.
    """

    surfaces: list[Surface]

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_inverse_pg_transform_kwargs(f.surfaces))
        return data
