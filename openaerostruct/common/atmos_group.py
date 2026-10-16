from typing import Any

from pydantic import model_validator

import om4.api as om

from openaerostruct.common.atmos_comp import AtmosComp
from openaerostruct.common.reynolds_comp import ReynoldsComp


def _atmos_kwargs():
    return {
        "subsystems": {
            "atmos": AtmosComp(
                promotes_inputs=["altitude", "Mach_number"],
                promotes_outputs=["T", "P", "rho", "speed_of_sound", "mu", "v"],
            ),
            "reynolds": ReynoldsComp(promotes_inputs=["rho", "mu", "v"], promotes_outputs=["re"]),
        }
    }


class AtmosGroup(om.Group):
    """Standard atmosphere properties and Reynolds number from altitude and Mach number."""

    @model_validator(mode="before")
    @classmethod
    def _build_from_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data.update(_atmos_kwargs())
        return data
