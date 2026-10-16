from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om
from om4.derivatives.complex_step import ComplexStep
from om4.derivatives.differentiator import Differentiator
from om4.utils.polymorphic import Polymorphic

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface


class ComputeThrustLoads(om.ExplicitComponent):
    """
    Compute the loads on the structure due to the thrust of engines.
    The current method adds loads and moments to all of the structural nodes, but
    the nodes closest to the engines receive proportionally larger loads.

    Parameters
    ----------
    point_mass_locations[n_point_masses, 3] : numpy array
        XYZ location for each point mass in the global frame.
    engine_thrusts[n_point_masses] : numpy array
        Actual magnitude of each engine thrust, in same order as point_mass_locations.
    nodes[ny, 3] : numpy array
        Flattened array with coordinates for each FEM node.

    Returns
    -------
    nodal_weightings[n_point_masses, ny] : numpy array
        The normalized weighting factor for each node. The closest nodes have the
        greatest weighting, while farther away nodes have less.
    loads_from_thrusts[ny, 6] : numpy array
        The actual loads array that will be added to the total loads array.
        This is cumulative and includes the forces and moments from all point masses.
    """

    surface: Surface
    differentiator: Polymorphic[Differentiator] | None = ComplexStep()
    check_partials_options: om.CheckPartialsOptions | None = om.CheckPartialsOptions(differentiator="fd")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        surface = f.surface
        ny = surface["mesh"].shape[1]
        n_point_masses = surface["n_point_masses"]
        d = VarDecl()

        d.add_input("point_mass_locations", val=np.zeros((n_point_masses, 3)), units="m")
        d.add_input("engine_thrusts", val=np.zeros((n_point_masses)), units="N")
        d.add_input("nodes", val=np.zeros((ny, 3)), units="m")

        d.add_output("nodal_weightings", val=np.zeros((n_point_masses, ny)))
        d.add_output(
            "loads_from_thrusts", val=np.zeros((ny, 6)), units="N"
        )  # WARNING!!! UNITS ARE A MIXTURE OF N & N*m

        d.declare_partials("*", "*")
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nodes = inputs["nodes"]
        ny = nodes.shape[0]
        n_point_masses = self.surface["n_point_masses"]
        dtype = np.result_type(nodes, inputs["point_mass_locations"], inputs["engine_thrusts"])
        nodal_weightings = np.zeros((n_point_masses, ny), dtype=dtype)
        loads_from_engine_thrusts = np.zeros((ny, 6), dtype=dtype)

        # Loop through each point mass location, incrementing idx
        for idx, point_mass_location in enumerate(inputs["point_mass_locations"]):
            # Get the vector between the nodes and the point mass location
            xyz_dist = point_mass_location - nodes

            # The y-distance between the nodes and the point mass location
            span_dist = xyz_dist[:, 1]

            # Compute the normalized inverse distance weightings for all of the
            # nodes. These weightings determine the amount of the force and
            # moment that each of the nodes receive.
            # nodal_weightings are scalars for each node.
            dist10 = span_dist**10  # TODO: investigate effect of power here
            inv_dist10 = 1 / (dist10 + 1e-10)
            nodal_weightings[idx, :] = inv_dist10 / np.sum(inv_dist10)

            # Create an array with the inverse distance weighted vectors in the
            # forward chordwise direction (negative x). Each node has a 3-vector
            # in the forward direction whose magnitude is based on nodal_weightings.
            directional_weightings = np.outer(nodal_weightings[idx, :], np.array([[-1.0, 0.0, 0.0]]))

            # Compute the perceived weight due to the point mass on each node in N
            weight_forces = directional_weightings * inputs["engine_thrusts"][idx]

            # Actually add the perceived weights to the load array
            loads_from_engine_thrusts[:, :3] += weight_forces

            # Compute the moments based on the Euclidean distance vectors from
            # the nodes to the point mass, crossed with the perceived weight
            moments = np.cross(xyz_dist, weight_forces)

            # Accumulate the moments to the load array
            loads_from_engine_thrusts[:, 3:] += moments

        outputs["nodal_weightings"] = nodal_weightings
        outputs["loads_from_thrusts"] = loads_from_engine_thrusts
