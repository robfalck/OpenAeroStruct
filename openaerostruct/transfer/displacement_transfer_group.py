from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.transfer.compute_transformation_matrix import ComputeTransformationMatrix
from openaerostruct.transfer.displacement_transfer import DisplacementTransfer
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _displacement_transfer_kwargs(surface: Surface) -> dict:
    allp = {"promotes_inputs": ["*"], "promotes_outputs": ["*"]}
    return {
        "subsystems": {
            "compute_transformation_matrix": ComputeTransformationMatrix(surface=surface, **allp),
            "displacement_transfer": DisplacementTransfer(surface=surface, **allp),
        }
    }


class DisplacementTransferGroup(om.Group):
    """
    These components take the displacements and rotations obtained by
    solving the FEM problem and applies them to the aerodynamic mesh
    to produce a deformed aerodynamic mesh.
    """

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_displacement_transfer_kwargs(f.surface))
        return data
