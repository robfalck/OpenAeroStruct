from typing import Any

from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

import om4.api as om
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from openaerostruct.structures.transform import Transform
from openaerostruct.structures.length import Length
from openaerostruct.structures.local_stiff import LocalStiff
from openaerostruct.structures.local_stiff_permuted import LocalStiffPermuted
from openaerostruct.structures.local_stiff_transformed import LocalStiffTransformed
from openaerostruct.utils.om4_utils import field_values
from openaerostruct.utils.surface import Surface


def _assemble_k_kwargs(surface: Surface) -> dict:
    allp = {"promotes_inputs": ["*"], "promotes_outputs": ["*"]}
    return {
        "subsystems": {
            "transform": Transform(surface=surface, **allp),
            "length": Length(surface=surface, **allp),
            "local_stiff": LocalStiff(surface=surface, **allp),
            "local_stiff_permuted": LocalStiffPermuted(surface=surface, **allp),
            "local_stiff_transformed": LocalStiffTransformed(surface=surface, **allp),
        }
    }


class AssembleKGroup(om.Group):
    """Assemble that there compact local stiffness matrix."""

    surface: Surface

    # Derived from `surface`; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        data.update(_assemble_k_kwargs(f.surface))
        return data
