from functools import cached_property
from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values, vlm_system_size
from openaerostruct.utils.surface import Surface



class VLMMtxRHSComp(om.ExplicitComponent):
    """
    Compute the total velocities at each of the evaluation points for every
    panel in the entire system. This is the sum of the freestream and induced
    velocities caused by the circulations.

    Parameters
    ----------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    vel_mtx[num_eval_points, nx - 1, ny - 1, 3] : numpy array
        The AIC matrix for the all lifting surfaces representing the aircraft.
        This has some sparsity pattern, but it is more dense than the FEM matrix
        and the entries have a wide range of magnitudes. One exists for each
        combination of surface name and evaluation points name.
    normals[nx-1, ny-1, 3] : numpy array
        The normal vector for each panel, computed as the cross of the two
        diagonals from the mesh points.

    Returns
    -------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.
    """

    surfaces: list[Surface]

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = vars(field_values(cls, data))
        d = VarDecl()
        _spec = data  # the setup code below may rebind `data`

        surfaces = f["surfaces"]

        system_size = 0

        # Loop through the surfaces to compute the total number of panels;
        # the system_size
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            system_size += (nx - 1) * (ny - 1)

        d.add_input("freestream_velocities", shape=(system_size, 3), units="m/s")
        d.add_output("mtx", shape=(system_size, system_size), units="1/m")
        d.add_output("rhs", shape=system_size, units="m/s")

        # Set up indicies arrays for sparse Jacobians
        vel_indices = np.arange(system_size * 3).reshape((system_size, 3))
        mtx_indices = np.arange(system_size * system_size).reshape((system_size, system_size))
        rhs_indices = np.arange(system_size)

        d.declare_partials(
            "rhs",
            "freestream_velocities",
            rows=np.einsum("i,j->ij", rhs_indices, np.ones(3, int)).flatten(),
            cols=vel_indices.flatten(),
        )

        ind_1 = 0
        ind_2 = 0

        # Loop through each surface to add inputs and set up derivatives.
        # We keep track of the surface's indices within the total system's
        # indices to access the matrix in the correct locations for the derivs.
        # This is because the AIC linear system has information for all surfaces
        # together.
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            # Get the correct names for each vel_mtx and normals, then
            # add them to the component
            vel_mtx_name = "{}_{}_vel_mtx".format(name, "coll_pts")
            normals_name = "{}_normals".format(name)

            d.add_input(vel_mtx_name, shape=(system_size, nx - 1, ny - 1, 3), units="1/m")
            d.add_input(normals_name, shape=(nx - 1, ny - 1, 3))

            velocities_indices = np.arange(system_size * num * 3).reshape((system_size, nx - 1, ny - 1, 3))
            normals_indices = np.arange(num * 3).reshape((num, 3))

            # Declare each set of partials based on the indices, ind_1 and ind_2
            d.declare_partials(
                "mtx",
                vel_mtx_name,
                rows=np.einsum("ij,k->ijk", mtx_indices[:, ind_1:ind_2], np.ones(3, int)).flatten(),
                cols=velocities_indices.flatten(),
            )
            d.declare_partials(
                "mtx",
                normals_name,
                rows=np.einsum("ij,k->ijk", mtx_indices[ind_1:ind_2, :], np.ones(3, int)).flatten(),
                cols=np.einsum("ik,j->ijk", normals_indices, np.ones(system_size, int)).flatten(),
            )
            d.declare_partials(
                "rhs",
                normals_name,
                rows=np.outer(rhs_indices[ind_1:ind_2], np.ones(3, int)).flatten(),
                cols=normals_indices.flatten(),
            )

            ind_1 += num

        return d.into(_spec)

    @cached_property
    def system_size(self) -> int:
        """Total number of VLM panels over all surfaces."""
        return vlm_system_size(self.surfaces)

    def _assemble(self, inputs):
        """
        Stack every surface's vel_mtx and normals into system-sized arrays.

        OM3 kept these as instance scratch arrays filled in compute and read back in
        compute_partials; recomputing them removes that dependence on call order.
        """
        surfaces = self.surfaces

        system_size = self.system_size
        mtx_n_n_3 = np.zeros((system_size, system_size, 3))
        normals_n_3 = np.zeros((system_size, 3))

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = "{}_{}_vel_mtx".format(name, "coll_pts")
            normals_name = "{}_normals".format(name)

            # Construct the full matrix and all of the lifting surfaces
            # together
            # TODO: This is not complex-safe
            mtx_n_n_3[:, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape((system_size, num, 3))
            normals_n_3[ind_1:ind_2, :] = inputs[normals_name].reshape((num, 3))

            ind_1 += num

        return mtx_n_n_3, normals_n_3

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtx_n_n_3, normals_n_3 = self._assemble(inputs)

        # Actually obtain the final matrix by multiplying through with the
        # normals. Also create the rhs based on v dot n.
        outputs["mtx"] = np.einsum("ijk,ik->ij", mtx_n_n_3, normals_n_3)
        outputs["rhs"] = -np.einsum("ij,ij->i", inputs["freestream_velocities"], normals_n_3)

    def compute_partials(self, inputs, partials):
        surfaces = self.surfaces
        mtx_n_n_3, normals_n_3 = self._assemble(inputs)

        system_size = self.system_size

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            vel_mtx_name = "{}_{}_vel_mtx".format(name, "coll_pts")
            normals_name = "{}_normals".format(name)

            partials["mtx", vel_mtx_name] = np.einsum(
                "ijk,ik->ijk",
                np.ones((system_size, num, 3)),
                normals_n_3,
            ).flatten()

            partials["mtx", normals_name] = mtx_n_n_3[ind_1:ind_2, :, :].flatten()

            partials["rhs", normals_name] = -inputs["freestream_velocities"][ind_1:ind_2, :].flatten()

            ind_1 += num

        partials["rhs", "freestream_velocities"] = -normals_n_3.flatten()
