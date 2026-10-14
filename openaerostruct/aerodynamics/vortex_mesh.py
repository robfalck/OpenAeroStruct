from typing import Any

import numpy as np
from pydantic import model_validator

import om4.api as om

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface



def _q12_data(nx, ny):
    """Constant partial values of the baseline mesh (quadrant 1) and its y-reflection (quadrant 2)."""
    data = np.concatenate(
        [
            0.75 * np.ones((nx - 1) * ny * 3),
            0.25 * np.ones((nx - 1) * ny * 3),
            np.ones(ny * 3),
        ]
    )  # back row,
    return np.concatenate(
        [
            data,
            0.75 * np.ones((nx - 1) * (ny - 1)),
            0.25 * np.ones((nx - 1) * (ny - 1)),
            np.ones(ny - 1),
            -0.75 * np.ones((nx - 1) * (ny - 1)),
            -0.25 * np.ones((nx - 1) * (ny - 1)),
            -np.ones(ny - 1),
            0.75 * np.ones((nx - 1) * (ny - 1)),
            0.25 * np.ones((nx - 1) * (ny - 1)),
            np.ones(ny - 1),
        ]
    )


class VortexMesh(om.ExplicitComponent):
    """
    Compute the vortex mesh based on the deformed aerodynamic mesh.

    Parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        We have a mesh for each lifting surface in the problem.
        That is, if we have both a wing and a tail surface, we will have both
        `wing_def_mesh` and `tail_def_mesh` as inputs.
    height_agl : scalar
        If ground effect is turned on, this input defines the height above
        the groud plane (defined from the origin 0,0,0)
    alpha : scalar
        If ground effect is turned on, this input defines the angular
        rotation of the ground plane

    Returns
    -------
    vortex_mesh[nx, ny, 3] : numpy array
        The actual aerodynamic mesh used in VLM calculations, where we look
        at the rings of the panels instead of the panels themselves. That is,
        this mesh coincides with the quarter-chord panel line, except for the
        final row, where it lines up with the trailing edge.
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

        # Because the vortex_mesh always comes from the deformed mesh in the
        # same way, the Jacobian is fully linear and can be set here instead
        # of doing compute_partials.
        # We do have to account for symmetry here to create a ghost mesh
        # by mirroring the symmetric mesh.

        any_ground_effect = False

        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]

            mesh_name = "{}_def_mesh".format(name)
            vortex_mesh_name = "{}_vortex_mesh".format(name)

            d.add_input(mesh_name, shape=(nx, ny, 3), units="m", tags=["mphys_coupling"])

            ground_effect = surface.get("groundplane", False)

            if ground_effect:
                if not any_ground_effect:
                    # only need to add the extra inputs once
                    any_ground_effect = True
                    d.add_input("height_agl", val=8000.0, units="m")
                    d.add_input("alpha", val=0.0 * np.pi / 180, units="rad", tags=["mphys_inputs"])

            if surface["symmetry"]:
                left_wing = abs(surface["mesh"][0, 0, 1]) > abs(surface["mesh"][0, -1, 1])
                if ground_effect:
                    d.add_output(vortex_mesh_name, shape=(2 * nx, ny * 2 - 1, 3), units="m")
                    # OM3 complex-stepped these; om4 computes them analytically (ai/OM4_NEEDS.md N-002)
                    d.declare_partials(vortex_mesh_name, ["alpha", "height_agl"])
                    mesh_indices = np.arange(nx * ny * 3).reshape((nx, ny, 3))
                    vor_indices = np.arange(2 * nx * (2 * ny - 1) * 3).reshape((2 * nx, (2 * ny - 1), 3))
                    if not left_wing:
                        vor_indices = vor_indices[:, ::-1, :]
                        mesh_indices = mesh_indices[:, ::-1, :]
                    quadrant_1_indices = vor_indices[:nx, :ny, :]
                    quadrant_2_indices = vor_indices[:nx, ny:, :]
                    quadrant_3_indices = vor_indices[nx:, :ny, :]
                    quadrant_4_indices = vor_indices[nx:, ny:, :]
                else:
                    # no groundplane
                    d.add_output(vortex_mesh_name, shape=(nx, ny * 2 - 1, 3), units="m")
                    mesh_indices = np.arange(nx * ny * 3).reshape((nx, ny, 3))
                    vor_indices = np.arange(nx * (2 * ny - 1) * 3).reshape((nx, (2 * ny - 1), 3))
                    if not left_wing:
                        vor_indices = vor_indices[:, ::-1, :]
                        mesh_indices = mesh_indices[:, ::-1, :]
                    quadrant_1_indices = vor_indices[:nx, :ny, :]
                    quadrant_2_indices = vor_indices[:nx, ny:, :]

                # quadrant 1 is just the baseline mesh
                rows = np.tile(quadrant_1_indices[:-1, :, :].flatten(), 2)
                rows = np.hstack((rows, quadrant_1_indices[-1, :, :].flatten()))
                cols = np.concatenate(
                    [
                        mesh_indices[:-1, :, :].flatten(),
                        mesh_indices[1:, :, :].flatten(),
                        mesh_indices[-1, :, :].flatten(),
                    ]
                )

                data = np.concatenate(
                    [
                        0.75 * np.ones((nx - 1) * ny * 3),
                        0.25 * np.ones((nx - 1) * ny * 3),
                        np.ones(ny * 3),
                    ]
                )  # back row,

                # quadrant 2 is the reflection of the baseline across the midline
                # need to build these piecewise xyz because of the midline reflection
                for dim3 in range(3):
                    rows = np.hstack((rows, np.tile(quadrant_2_indices[:-1, :, dim3].flatten(), 2)))
                    rows = np.hstack((rows, quadrant_2_indices[-1, :, dim3].flatten()))
                    cols = np.concatenate(
                        [
                            cols,
                            mesh_indices[:-1, :-1, dim3][:, ::-1].flatten(),
                            mesh_indices[1:, :-1, dim3][:, ::-1].flatten(),
                            mesh_indices[-1, :-1, dim3][::-1].flatten(),
                        ]
                    )

                data = np.concatenate(
                    [
                        data,
                        0.75 * np.ones((nx - 1) * (ny - 1)),
                        0.25 * np.ones((nx - 1) * (ny - 1)),
                        np.ones(ny - 1),
                        -0.75 * np.ones((nx - 1) * (ny - 1)),
                        -0.25 * np.ones((nx - 1) * (ny - 1)),
                        -np.ones(ny - 1),
                        0.75 * np.ones((nx - 1) * (ny - 1)),
                        0.25 * np.ones((nx - 1) * (ny - 1)),
                        np.ones(ny - 1),
                    ]
                )

                if ground_effect:
                    # these reflections (across the groundplane) are more complex because of the alpha rotation
                    # which means that the x and z points of the reflected mesh depend on BOTH the x and z points of the initial mesh
                    # y only depends on y as usual

                    # third quadrant dependencies (x on x, y on y, z on z, x on z, z on x)
                    list_of_deps = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)]
                    for dep_of, dep_on in list_of_deps:
                        rows = np.hstack((rows, np.tile(quadrant_3_indices[:-1, :, dep_of].flatten(), 2)))
                        rows = np.hstack((rows, quadrant_3_indices[-1, :, dep_of].flatten()))
                        cols = np.concatenate(
                            [
                                cols,
                                mesh_indices[:-1, :, dep_on].flatten(),
                                mesh_indices[1:, :, dep_on].flatten(),
                                mesh_indices[-1, :, dep_on].flatten(),
                            ]
                        )

                    # fourth quadrant dependencies (x on x, y on y, z on z, x on z, z on x)
                    for dep_of, dep_on in list_of_deps:
                        rows = np.hstack((rows, np.tile(quadrant_4_indices[:-1, :, dep_of].flatten(), 2)))
                        rows = np.hstack((rows, quadrant_4_indices[-1, :, dep_of].flatten()))
                        cols = np.concatenate(
                            [
                                cols,
                                mesh_indices[:-1, :-1, dep_on][:, ::-1].flatten(),
                                mesh_indices[1:, :-1, dep_on][:, ::-1].flatten(),
                                mesh_indices[-1, :-1, dep_on][::-1].flatten(),
                            ]
                        )

                    # can't declare constant partials because these depend on alpha (and h?)
                    d.declare_partials(vortex_mesh_name, mesh_name, rows=rows, cols=cols)

                else:
                    # no groundplane, constant partial values
                    d.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)

            else:
                if ground_effect:
                    raise ValueError("Ground effect is not supported without symmetry turned on")

                d.add_output(vortex_mesh_name, shape=(nx, ny, 3), units="m")

                mesh_indices = np.arange(nx * ny * 3).reshape((nx, ny, 3))

                rows = np.tile(mesh_indices[: (nx - 1), :, :].flatten(), 2)
                rows = np.hstack((rows, mesh_indices[-1, :, :].flatten()))
                cols = np.concatenate(
                    [
                        mesh_indices[:-1, :, :].flatten(),
                        mesh_indices[1:, :, :].flatten(),
                        mesh_indices[-1, :, :].flatten(),
                    ]
                )

                data = np.concatenate(
                    [
                        0.75 * np.ones((nx - 1) * ny * 3),
                        0.25 * np.ones((nx - 1) * ny * 3),
                        np.ones(ny * 3),  # back row
                    ]
                )

                d.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)
        return d.into(_spec)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        surfaces = self.surfaces

        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            ground_effect = surface.get("groundplane", False)
            left_wing = abs(surface["mesh"][0, 0, 1]) > abs(surface["mesh"][0, -1, 1])

            mesh_name = "{}_def_mesh".format(name)
            vortex_mesh_name = "{}_vortex_mesh".format(name)
            if not ground_effect:
                if surface["symmetry"]:
                    mesh = np.zeros((nx, ny * 2 - 1, 3), dtype=type(inputs[mesh_name][0, 0, 0]))
                    # Check if the wing is a left or right wing.
                    # Regardless, the "y" node ordering must always go from left to right
                    # for the aic matrix procedure to work correctly
                    if left_wing:
                        mesh[:, :ny, :] = inputs[mesh_name]
                        # indices are numbered from tip to centerline
                        #  reflection is all but midpoint in rev order
                        mesh[:, ny:, :] = inputs[mesh_name][:, :-1, :][:, ::-1, :]
                        mesh[:, ny:, 1] *= -1.0
                    else:
                        mesh[:, ny - 1 :, :] = inputs[mesh_name]
                        # indices are numbered from centerline to tip
                        #  reflection is all points in rev order
                        mesh[:, : ny - 1, :] = inputs[mesh_name][:, 1:, :][:, ::-1, :]
                        mesh[:, : ny - 1, 1] *= -1.0
                else:
                    mesh = inputs[mesh_name]

                # all but the last station are moved to the quarterchord point
                outputs[vortex_mesh_name][:-1, :, :] = 0.75 * mesh[:-1, :, :] + 0.25 * mesh[1:, :, :]
                # the last one is coincident
                outputs[vortex_mesh_name][-1, :, :] = mesh[-1, :, :]
            else:
                # symmetric in y plus ground plane using the first dimension
                mesh = np.zeros((2 * nx, ny * 2 - 1, 3), dtype=type(inputs[mesh_name][0, 0, 0]))

                if left_wing:
                    mesh[:nx, :ny, :] = inputs[mesh_name]
                    # indices are numbered from tip to centerline
                    #  reflection is all but midpoint in rev order
                    mesh[:nx, ny:, :] = inputs[mesh_name][:, :-1, :][:, ::-1, :]
                    mesh[:nx, ny:, 1] *= -1.0
                else:
                    mesh[:nx, ny - 1 :, :] = inputs[mesh_name]
                    # indices are numbered from centerline to tip
                    #  reflection is all points in rev order
                    mesh[:nx, : ny - 1, :] = inputs[mesh_name][:, 1:, :][:, ::-1, :]
                    mesh[:nx, : ny - 1, 1] *= -1.0

                alpha = inputs["alpha"][0]
                plane_normal = np.array([np.sin(alpha), 0.0, -np.cos(alpha)]).reshape((1, 1, 3))
                plane_point = np.zeros((1, 1, 3)) + plane_normal * inputs["height_agl"]

                # reflect about the ground plane
                # plane is defined parallel to the free stream and height_agl from the origin 0 0 0
                v = mesh[:nx, :, :] - plane_point
                temp = np.inner(v, plane_normal).squeeze()[:, :, np.newaxis]
                v_par = temp * plane_normal
                mesh[nx:, :, :] = mesh[:nx, :, :] - 2 * v_par

                outputs[vortex_mesh_name][: nx - 1, :, :] = 0.75 * mesh[: nx - 1, :, :] + 0.25 * mesh[1:nx, :, :]
                outputs[vortex_mesh_name][nx - 1, :, :] = mesh[nx - 1, :, :]
                outputs[vortex_mesh_name][nx:-1, :, :] = 0.75 * mesh[nx:-1, :, :] + 0.25 * mesh[nx + 1 :, :, :]
                outputs[vortex_mesh_name][-1, :, :] = mesh[-1, :, :]

    @staticmethod
    def _y_mirrored_mesh(surface, def_mesh):
        """The surface plus its reflection across the symmetry plane, ordered left to right."""
        nx, ny = def_mesh.shape[:2]
        left_wing = abs(surface["mesh"][0, 0, 1]) > abs(surface["mesh"][0, -1, 1])
        mesh = np.zeros((nx, ny * 2 - 1, 3), dtype=def_mesh.dtype)
        if left_wing:
            mesh[:, :ny, :] = def_mesh
            mesh[:, ny:, :] = def_mesh[:, :-1, :][:, ::-1, :]
            mesh[:, ny:, 1] *= -1.0
        else:
            mesh[:, ny - 1 :, :] = def_mesh
            mesh[:, : ny - 1, :] = def_mesh[:, 1:, :][:, ::-1, :]
            mesh[:, : ny - 1, 1] *= -1.0
        return mesh

    def compute_partials(self, inputs, J):
        surfaces = self.surfaces
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface["name"]
            ground_effect = surface.get("groundplane", False)

            mesh_name = "{}_def_mesh".format(name)
            vortex_mesh_name = "{}_vortex_mesh".format(name)
            if not ground_effect:
                # if ground effect is not enabled the derivatives are constant
                # and this method need nto be called
                pass
            else:
                data = _q12_data(nx, ny)
                # we've already figured out the partials for quadrants 1 and 2
                # quandrants 3 and 4 are the ground plane reflections which
                # depend on angle of attack so they need to be computed each time

                # first comes quadrant 3
                # x on x, y on y, z on z, x on z, z on x is the order
                alpha = inputs["alpha"]
                x_on_x_const = 1 - 2 * np.sin(alpha) ** 2
                z_on_z_const = 1 - 2 * np.cos(alpha) ** 2
                x_on_z_const = 2 * np.sin(alpha) * np.cos(alpha)
                z_on_x_const = 2 * np.sin(alpha) * np.cos(alpha)

                data = np.concatenate(
                    [
                        data,
                        # x on x
                        x_on_x_const * 0.75 * np.ones((nx - 1) * ny),
                        x_on_x_const * 0.25 * np.ones((nx - 1) * ny),
                        x_on_x_const * np.ones(ny),
                        # y on y
                        0.75 * np.ones((nx - 1) * ny),
                        0.25 * np.ones((nx - 1) * ny),
                        np.ones(ny),
                        # z on z
                        z_on_z_const * 0.75 * np.ones((nx - 1) * ny),
                        z_on_z_const * 0.25 * np.ones((nx - 1) * ny),
                        z_on_z_const * np.ones(ny),
                        # x on z
                        x_on_z_const * 0.75 * np.ones((nx - 1) * ny),
                        x_on_z_const * 0.25 * np.ones((nx - 1) * ny),
                        x_on_z_const * np.ones(ny),
                        # z on x
                        z_on_x_const * 0.75 * np.ones((nx - 1) * ny),
                        z_on_x_const * 0.25 * np.ones((nx - 1) * ny),
                        z_on_x_const * np.ones(ny),
                    ]
                )

                # now quadrant 4 with different dims and reflected y coords

                data = np.concatenate(
                    [
                        data,
                        # x on x
                        x_on_x_const * 0.75 * np.ones((nx - 1) * (ny - 1)),
                        x_on_x_const * 0.25 * np.ones((nx - 1) * (ny - 1)),
                        x_on_x_const * np.ones((ny - 1)),
                        # y on y
                        -0.75 * np.ones((nx - 1) * (ny - 1)),
                        -0.25 * np.ones((nx - 1) * (ny - 1)),
                        -np.ones((ny - 1)),
                        # z on z
                        z_on_z_const * 0.75 * np.ones((nx - 1) * (ny - 1)),
                        z_on_z_const * 0.25 * np.ones((nx - 1) * (ny - 1)),
                        z_on_z_const * np.ones((ny - 1)),
                        # x on z
                        x_on_z_const * 0.75 * np.ones((nx - 1) * (ny - 1)),
                        x_on_z_const * 0.25 * np.ones((nx - 1) * (ny - 1)),
                        x_on_z_const * np.ones((ny - 1)),
                        # z on x
                        z_on_x_const * 0.75 * np.ones((nx - 1) * (ny - 1)),
                        z_on_x_const * 0.25 * np.ones((nx - 1) * (ny - 1)),
                        z_on_x_const * np.ones((ny - 1)),
                    ]
                )

                J[vortex_mesh_name, mesh_name] = data

                # d vortex_mesh / d (alpha, height_agl).  The ground reflection of a point m is
                # m' = m - 2 (m.n - h) n, with n = (sin a, 0, -cos a) and h = height_agl, so
                # dm'/dh = 2 n and dm'/da = -2 ((m.dn) n + (m.n - h) dn), dn = (cos a, 0, sin a).
                # Only the reflected half (rows nx:) depends on them.
                a = inputs["alpha"][0]
                h = inputs["height_agl"][0]
                n = np.array([np.sin(a), 0.0, -np.cos(a)])
                dn = np.array([np.cos(a), 0.0, np.sin(a)])
                m = self._y_mirrored_mesh(surface, inputs[mesh_name])
                m_dot_n = np.einsum("ijk,k->ij", m, n)[:, :, np.newaxis]
                m_dot_dn = np.einsum("ijk,k->ij", m, dn)[:, :, np.newaxis]
                dref_da = -2.0 * (m_dot_dn * n + (m_dot_n - h) * dn)
                dref_dh = np.broadcast_to(2.0 * n, m.shape)
                for wrt, dref in (("alpha", dref_da), ("height_agl", dref_dh)):
                    dvort = np.zeros((2 * nx, 2 * ny - 1, 3))
                    dvort[nx:-1] = 0.75 * dref[:-1] + 0.25 * dref[1:]
                    dvort[-1] = dref[-1]
                    J[vortex_mesh_name, wrt] = dvort.reshape((-1, 1))
