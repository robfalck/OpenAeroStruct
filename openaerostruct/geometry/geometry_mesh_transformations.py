"""A set of components that manipulate geometry mesh
based on high-level design parameters.

om4 port notes: OM3 options are fields; variables and sparsity patterns are built in a
before-validator from module-level ``_*_pattern`` functions; ``compute_partials`` fills
each subjac in a local array and assigns it once (om4 subjacs are write-only).
"""

from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

import om4.api as om
from om4.utils.types import FloatOrNpArray

from openaerostruct.utils.om4_utils import field_values


MeshShape = tuple[int, int, int]


def _scalar(val) -> np.ndarray:
    """OM3 scalar variables have shape (1,)."""
    return np.atleast_1d(np.asarray(val, dtype=float))


def _ref_axis(mesh, ref_axis_pos):
    return ref_axis_pos * mesh[-1] + (1 - ref_axis_pos) * mesh[0]


class Taper(om.ExplicitComponent):
    """
    Manipulate the mesh by altering the spanwise chord linearly to produce a tapered wing.

    Taper is applied around the reference axis line, which is the quarter-chord by default.

    Parameters
    ----------
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the tapered aerodynamic surface.
    """

    val: float = Field(description="Initial value for the taper ratio.")
    mesh: FloatOrNpArray = Field(description="Nodal mesh defining the initial aerodynamic surface.")
    symmetry: bool = Field(default=False, description="True if surface is reflected about y=0 plane.")
    ref_axis_pos: float = Field(default=0.25, description="Fraction of the chord to use as the reference axis.")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh = np.asarray(f.mesh, dtype=float)
        data["inputs"] = {"taper": om.InputVar(val=_scalar(f.val), units=None)}
        data["outputs"] = {"mesh": om.OutputVar(val=mesh.copy(), units="m")}
        data["partials"] = [om.PartialsSpec(of="*", wrt="*")]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mesh = self.mesh
        symmetry = self.symmetry
        taper_ratio = inputs["taper"][0]

        # Get mesh parameters and the quarter-chord
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)
        x = ref_axis[:, 1]

        # Spanwise(j) index of wing centerline
        n_sym = (len(x) + 1) // 2 - 1

        # If symmetric, solve for the correct taper ratio, which is a linear
        # interpolation problem (assume symmetry axis is not necessarily at y = 0)
        if symmetry:
            xp = np.array([x[0], x[-1]])
            fp = np.array([taper_ratio, 1.0])

        # Otherwise, we set up an interpolation problem for the entire wing, which
        # consists of two linear segments (assume symmetry axis is not necessarily at y = 0)
        else:
            xp = np.array([x[0], x[n_sym], x[-1]])
            fp = np.array([taper_ratio, 1.0, taper_ratio])

        # Interpolate over quarter chord line to compute the taper at each spanwise stations
        taper = np.interp(x, xp, fp)

        # Broadcast taper array over the mesh along spanwise(j) index
        outputs["mesh"] = np.einsum("ijk,j->ijk", mesh - ref_axis, taper) + ref_axis

    def compute_partials(self, inputs, partials):
        mesh = self.mesh
        symmetry = self.symmetry

        ref_axis = _ref_axis(mesh, self.ref_axis_pos)
        x = ref_axis[:, 1]

        # Spanwise(j) index of wing centerline
        n_sym = (len(x) + 1) // 2 - 1

        # Derivative implementation that allows for taper_ratio = 1
        if symmetry:
            span = x[-1] - x[0]
            dy = x - x[0]
            # Derivative of the linear interpolation wrt the end point (the taper ratio)
            dtaper = np.ones(len(x)) + (-dy / span)
        else:
            span1 = x[n_sym] - x[0]
            dy1 = x[: n_sym + 1] - x[0]
            dtaper1 = np.ones(n_sym + 1) + (-dy1 / span1)

            span2 = x[-1] - x[n_sym]
            dy2 = x[n_sym + 1 :] - x[n_sym]
            dtaper2 = dy2 / span2

            dtaper = np.concatenate([dtaper1, dtaper2])

        partials["mesh", "taper"] = np.einsum("ijk, j->ijk", mesh - ref_axis, dtaper).reshape((-1, 1))


def _scale_x_pattern(mesh_shape):
    nx, ny, _ = mesh_shape
    nn = nx * ny * 3

    # d mesh / d chord: every mesh entry is sensitive to the chord at its spanwise station
    chord_rows = np.arange(nn)
    chord_cols = np.tile(np.repeat(np.arange(ny), 3), nx)

    # d mesh / d in_mesh: main diagonal plus the ref_axis (LE and TE) contributions
    p_rows = np.arange(nn)
    te_rows = np.arange((nx - 1) * ny * 3)
    le_rows = te_rows + ny * 3
    le_cols = np.tile(np.arange(3 * ny), nx - 1)
    te_cols = le_cols + ny * 3 * (nx - 1)
    rows = np.concatenate([p_rows, te_rows, le_rows])
    cols = np.concatenate([p_rows, te_cols, le_cols])
    return (chord_rows, chord_cols), (rows, cols)


class ScaleX(om.ExplicitComponent):
    """
    Manipulate the mesh by scaling the chords (x-coordinates) along the span about the reference axis.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    chord[ny] : numpy array
        Spanwise distribution of the chord scaler.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    val: FloatOrNpArray = Field(description="Initial value for chord lengths.")
    mesh_shape: MeshShape = Field(description="Mesh shape (nx, ny, 3).")
    ref_axis_pos: float = Field(default=0.25, description="Fraction of the chord to use as the reference axis.")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh_shape = tuple(f.mesh_shape)
        (c_rows, c_cols), (m_rows, m_cols) = _scale_x_pattern(mesh_shape)
        data["inputs"] = {
            "chord": om.InputVar(val=np.asarray(f.val, dtype=float), units=None),
            "in_mesh": om.InputVar(val=np.ones(mesh_shape), units="m"),
        }
        data["outputs"] = {"mesh": om.OutputVar(val=np.ones(mesh_shape), units="m")}
        data["partials"] = [
            om.PartialsSpec(of="mesh", wrt="chord", rows=c_rows, cols=c_cols),
            om.PartialsSpec(of="mesh", wrt="in_mesh", rows=m_rows, cols=m_cols),
        ]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["chord"]
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)
        outputs["mesh"] = np.einsum("ijk,j->ijk", mesh - ref_axis, chord_dist) + ref_axis

    def compute_partials(self, inputs, partials):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["chord"]
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)

        # The derivative wrt chord is the mesh itself (offset to ref_axis)
        partials["mesh", "chord"] = (mesh - ref_axis).flatten()

        nx, ny, _ = mesh.shape
        nn = nx * ny * 3
        nnq = (nx - 1) * ny * 3
        d = np.zeros(nn + 2 * nnq)

        # Diagonal: chord_dist broadcast over each mesh row
        d_mesh = np.einsum("i,ij->ij", chord_dist, np.ones((ny, 3))).flatten()
        d[:nn] = np.tile(d_mesh, nx)

        # (1 - chord_dist) broadcast onto a single row of the mesh, used by all ref_axis terms
        d_qc = (np.einsum("ij,i->ij", np.ones((ny, 3)), 1.0 - chord_dist)).flatten()

        # Off-diagonal: non-TE rows to the TE contribution, non-LE rows to the LE contribution
        d[nn : nn + nnq] = np.tile(self.ref_axis_pos * d_qc, nx - 1)
        d[nn + nnq :] = np.tile((1 - self.ref_axis_pos) * d_qc, nx - 1)

        # ref_axis contributions on the main diagonal for the TE and LE rows themselves
        nnr = ny * 3
        d[nn - nnr : nn] += self.ref_axis_pos * d_qc
        d[:nnr] += (1 - self.ref_axis_pos) * d_qc

        partials["mesh", "in_mesh"] = d


def _sweep_like_pattern(mesh_shape, symmetry, coord):
    """
    Sparsity of Sweep (coord=0, x) and Dihedral (coord=2, z) wrt their angle and in_mesh.

    The two components share the same structure, differing only in which coordinate moves.
    """
    nx, ny, _ = mesh_shape
    nn = nx * ny

    # The moved coordinate of every point is sensitive to the scalar angle (col 0)
    angle_rows = 3 * np.arange(nn) + coord
    angle_cols = np.zeros(nn, int)

    nn = nx * ny * 3
    n_rows = np.arange(nn)

    if symmetry:
        # y-coordinate index of the symmetry plane leading edge
        y_cp = ny * 3 - 2
        sym_cols = np.tile(y_cp, nx * (ny - 1))
        sym_rows = np.tile(3 * np.arange(ny - 1) + coord, nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)
        span_cols = np.tile(3 * np.arange(ny - 1) + 1, nx)
    else:
        # y-coordinate of the center line leading edge
        y_cp = 3 * (ny + 1) // 2 - 2
        n_sym = (ny - 1) // 2
        sym_row = np.tile(3 * np.arange(n_sym) + coord, 2) + np.repeat([0, 3 * (n_sym + 1)], n_sym)
        sym_rows = np.tile(sym_row, nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)
        sym_col = np.tile(y_cp, n_sym)
        span_col1 = 3 * np.arange(n_sym) + 1
        span_col2 = 3 * np.arange(n_sym) + 4 + 3 * n_sym
        # Swap columns on the reflected side so + and - tan entries are grouped together
        sym_cols = np.tile(np.concatenate([sym_col, span_col2]), nx)
        span_cols = np.tile(np.concatenate([span_col1, sym_col]), nx)

    rows = np.concatenate([n_rows, sym_rows, sym_rows])
    cols = np.concatenate([n_rows, sym_cols, span_cols])
    return (angle_rows, angle_cols), (rows, cols)


def _sweep_like_offsets(le, ny, tan_theta, symmetry):
    """Shift of the moved coordinate at each spanwise station for a shearing angle."""
    if symmetry:
        y0 = le[-1, 1]
        return -(le[:, 1] - y0) * tan_theta
    ny2 = (ny - 1) // 2
    y0 = le[ny2, 1]
    right = (le[ny2:, 1] - y0) * tan_theta
    left = -(le[:ny2, 1] - y0) * tan_theta
    return np.hstack((left, right))


class _SweepLike(om.ExplicitComponent):
    """Shared implementation of Sweep and Dihedral (shearing by an angle about the root)."""

    val: float = Field(description="Initial value for the angle in degrees.")
    mesh_shape: MeshShape = Field(description="Mesh shape (nx, ny, 3).")
    symmetry: bool = Field(default=False, description="True if surface is reflected about y=0 plane.")

    _angle_name: ClassVar[str] = ""
    _coord: ClassVar[int] = 0

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh_shape = tuple(f.mesh_shape)
        (a_rows, a_cols), (m_rows, m_cols) = _sweep_like_pattern(mesh_shape, f.symmetry, cls._coord)
        data["inputs"] = {
            cls._angle_name: om.InputVar(val=_scalar(f.val), units="deg"),
            "in_mesh": om.InputVar(val=np.ones(mesh_shape), units="m"),
        }
        data["outputs"] = {"mesh": om.OutputVar(val=np.ones(mesh_shape), units="m")}
        data["partials"] = [
            om.PartialsSpec(of="mesh", wrt=cls._angle_name, rows=a_rows, cols=a_cols),
            om.PartialsSpec(of="mesh", wrt="in_mesh", rows=m_rows, cols=m_cols),
        ]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        angle = inputs[self._angle_name][0]
        mesh = inputs["in_mesh"]
        _, ny, _ = mesh.shape
        tan_theta = np.tan(np.pi / 180 * angle)
        delta = _sweep_like_offsets(mesh[0], ny, tan_theta, self.symmetry)

        outputs["mesh"][:] = mesh
        outputs["mesh"][:, :, self._coord] += delta

    def compute_partials(self, inputs, partials):
        angle = inputs[self._angle_name][0]
        mesh = inputs["in_mesh"]
        nx, ny, _ = mesh.shape
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * angle)

        # Derivative of tan(theta) wrt theta, times distance from the center of the wing
        dtan_dtheta = p180 / np.cos(p180 * angle) ** 2
        d_dtheta = _sweep_like_offsets(mesh[0], ny, dtan_dtheta, self.symmetry)
        partials["mesh", self._angle_name] = np.tile(d_dtheta, nx)

        # Diagonal passes in_mesh through; off-diagonal +tan / -tan for spanwise station sensitivity
        nn = nx * ny * 3
        nn2 = nx * (ny - 1)
        d = np.empty(nn + 2 * nn2)
        d[:nn] = 1.0
        d[nn : nn + nn2] = tan_theta
        d[nn + nn2 :] = -tan_theta
        partials["mesh", "in_mesh"] = d


class Sweep(_SweepLike):
    """
    Manipulate the mesh by applying shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    sweep : float
        Shearing sweep angle in degrees.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the swept aerodynamic surface.
    """

    _angle_name: ClassVar[str] = "sweep"
    _coord: ClassVar[int] = 0


class Dihedral(_SweepLike):
    """
    Manipulate the mesh by applying a dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    dihedral : float
        Dihedral angle in degrees.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the aerodynamic surface with dihedral angle.
    """

    _angle_name: ClassVar[str] = "dihedral"
    _coord: ClassVar[int] = 2


class _Shear(om.ExplicitComponent):
    """Shared implementation of ShearX/Y/Z: translate one coordinate by a spanwise distribution."""

    val: FloatOrNpArray = Field(description="Initial value for the shear distribution.")
    mesh_shape: MeshShape = Field(description="Mesh shape (nx, ny, 3).")

    _shear_name: ClassVar[str] = ""
    _coord: ClassVar[int] = 0

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh_shape = tuple(f.mesh_shape)
        nx, ny, _ = mesh_shape
        nn = nx * ny
        nn3 = nn * 3
        data["inputs"] = {
            cls._shear_name: om.InputVar(val=np.asarray(f.val, dtype=float), units="m"),
            "in_mesh": om.InputVar(val=np.ones(mesh_shape), units="m"),
        }
        data["outputs"] = {"mesh": om.OutputVar(val=np.ones(mesh_shape), units="m")}
        data["partials"] = [
            # The shifted coordinate of every point maps one to one to its spanwise station
            om.PartialsSpec(
                of="mesh",
                wrt=cls._shear_name,
                rows=3 * np.arange(nn) + cls._coord,
                cols=np.tile(np.arange(ny), nx),
                val=np.ones(nn),
            ),
            # Derivative of mesh wrt in_mesh is just identity
            om.PartialsSpec(of="mesh", wrt="in_mesh", rows=np.arange(nn3), cols=np.arange(nn3), val=np.ones(nn3)),
        ]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["mesh"][:] = inputs["in_mesh"]
        outputs["mesh"][:, :, self._coord] += inputs[self._shear_name]


class ShearX(_Shear):
    """
    Manipulate the mesh by shearing the wing in the x direction (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    xshear[ny] : numpy array
        Distance to translate wing in x direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    _shear_name: ClassVar[str] = "xshear"
    _coord: ClassVar[int] = 0


class ShearY(_Shear):
    """
    Manipulate the mesh by shearing the wing in the y direction.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    yshear[ny] : numpy array
        Distance to translate wing in y direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    _shear_name: ClassVar[str] = "yshear"
    _coord: ClassVar[int] = 1


class ShearZ(_Shear):
    """
    Manipulate the mesh by shearing the wing in the z direction (distributed dihedral).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    zshear[ny] : numpy array
        Distance to translate wing in z direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    _shear_name: ClassVar[str] = "zshear"
    _coord: ClassVar[int] = 2


def _stretch_pattern(mesh_shape):
    nx, ny, _ = mesh_shape
    nn = nx * ny

    # All y components of every mesh point are sensitive to the scalar span (col 0)
    span_rows = 3 * np.arange(nn) + 1
    span_cols = np.zeros(nn, int)

    # x and z on the diagonal are identity (z is x offset by 2)
    xz_diag = 3 * np.arange(nn)

    # y at the four corners of the mesh: le (tip, root) and te (tip, root)
    i_le0 = 1
    i_le1 = ny * 3 - 2
    i_te0 = (nx - 1) * ny * 3 + 1
    i_te1 = nn * 3 - 2
    rows_4c = np.tile(3 * np.arange(nn) + 1, 4)
    cols_4c = np.concatenate([np.tile(i_le0, nn), np.tile(i_le1, nn), np.tile(i_te0, nn), np.tile(i_te1, nn)])

    # y diagonal stripes for the rest of the mesh, two contributions (LE and TE)
    base = 3 * np.arange(1, ny - 1) + 1
    row_dg = np.tile(base, nx) + np.repeat(ny * 3 * np.arange(nx), ny - 2)
    rows_dg = np.tile(row_dg, 2)
    col_dg = np.tile(base, nx)
    cols_dg = np.concatenate([col_dg, col_dg + 3 * ny * (nx - 1)])

    rows = np.concatenate([xz_diag, xz_diag + 2, rows_4c, rows_dg])
    cols = np.concatenate([xz_diag, xz_diag + 2, cols_4c, cols_dg])
    return (span_rows, span_cols), (rows, cols)


class Stretch(om.ExplicitComponent):
    """
    Manipulate the mesh by stretching it in the spanwise direction to reach a specified span.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    span : float
        Full span of the surface (halved internally when symmetric).

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the stretched aerodynamic surface.
    """

    val: float = Field(description="Initial value for span.")
    mesh_shape: MeshShape = Field(description="Mesh shape (nx, ny, 3).")
    symmetry: bool = Field(default=False, description="True if surface is reflected about y=0 plane.")
    ref_axis_pos: float = Field(default=0.25, description="Fraction of the chord to use as the reference axis.")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh_shape = tuple(f.mesh_shape)
        (s_rows, s_cols), (m_rows, m_cols) = _stretch_pattern(mesh_shape)
        data["inputs"] = {
            "span": om.InputVar(val=_scalar(f.val), units="m"),
            "in_mesh": om.InputVar(val=np.ones(mesh_shape), units="m"),
        }
        data["outputs"] = {"mesh": om.OutputVar(val=np.ones(mesh_shape), units="m")}
        data["partials"] = [
            om.PartialsSpec(of="mesh", wrt="span", rows=s_rows, cols=s_cols),
            om.PartialsSpec(of="mesh", wrt="in_mesh", rows=m_rows, cols=m_cols),
        ]
        return data

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        span = inputs["span"][0]
        mesh = inputs["in_mesh"]
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)

        # The user always deals with the full span, so halve it when symmetric.
        if self.symmetry:
            span /= 2.0

        # Scale the y-coordinates so the reference axis spans the desired length
        prev_span = ref_axis[-1, 1] - ref_axis[0, 1]
        s = ref_axis[:, 1] / prev_span

        outputs["mesh"][:] = mesh
        outputs["mesh"][:, :, 1] = s * span

    def compute_partials(self, inputs, partials):
        span = inputs["span"][0]
        mesh = inputs["in_mesh"]
        nx, ny, _ = mesh.shape
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)
        rap = self.ref_axis_pos

        if self.symmetry:
            span /= 2.0

        prev_span = ref_axis[-1, 1] - ref_axis[0, 1]
        s = ref_axis[:, 1] / prev_span

        if self.symmetry:
            partials["mesh", "span"] = np.tile(0.5 * s, nx)
        else:
            partials["mesh", "span"] = np.tile(s, nx)

        # derivative of s wrt the prev_span, and wrt the ref axis end points
        d_prev_span = -ref_axis[:, 1] / prev_span**2
        d_prev_span_qc0 = np.zeros((ny,))
        d_prev_span_qc1 = np.zeros((ny,))
        d_prev_span_qc0[0] = d_prev_span_qc1[-1] = 1.0 / prev_span

        nn = nx * ny * 2
        nn2 = nx * ny
        nn3 = nn + nn2 * 2
        nn4 = nn3 + nn2
        nn5 = nn4 + nn2
        nn6 = nn5 + nx * (ny - 2)
        d = np.empty(nn6 + nx * (ny - 2))

        # x and z diagonals
        d[:nn] = 1.0
        # LE tip, LE root, TE tip, TE root
        d[nn : nn + nn2] = np.tile(-(1 - rap) * span * (d_prev_span - d_prev_span_qc0), nx)
        d[nn + nn2 : nn3] = np.tile((1 - rap) * span * (d_prev_span + d_prev_span_qc1), nx)
        d[nn3:nn4] = np.tile(-rap * span * (d_prev_span - d_prev_span_qc0), nx)
        d[nn4:nn5] = np.tile(rap * span * (d_prev_span + d_prev_span_qc1), nx)
        # Non-corner LE and TE
        d[nn5:nn6] = (1 - rap) * span / prev_span
        d[nn6:] = rap * span / prev_span

        partials["mesh", "in_mesh"] = d


def _rotate_pattern(mesh_shape, symmetry):
    nx, ny, _ = mesh_shape
    nn = nx * ny * 3

    # d mesh / d twist: each spanwise station is sensitive to the twist at that station
    twist_rows = np.arange(nn)
    twist_cols = np.tile(np.repeat(np.arange(ny), 3), nx)

    # Each entry of a mesh point is sensitive to each entry of the same in_mesh point
    row_base = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    col_base = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    # Diagonal
    nn = nx * ny
    dg_row = np.tile(row_base, nn) + np.repeat(3 * np.arange(nn), 9)
    dg_col = np.tile(col_base, nn) + np.repeat(3 * np.arange(nn), 9)

    # Leading and trailing edge on-diagonal terms
    row_base_y = np.tile(row_base, ny) + np.repeat(3 * np.arange(ny), 9)
    col_base_y = np.tile(col_base, ny) + np.repeat(3 * np.arange(ny), 9)
    nn2 = 3 * ny
    te_dg_row = np.tile(row_base_y, nx - 1) + np.repeat(nn2 * np.arange(nx - 1), 9 * ny)
    le_dg_col = np.tile(col_base_y, nx - 1)
    le_dg_row = te_dg_row + nn2
    te_dg_col = le_dg_col + 3 * ny * (nx - 1)

    # Leading and trailing edge off-diagonal terms
    if symmetry:
        row_base_y = np.tile(row_base, ny - 1) + np.repeat(3 * np.arange(ny - 1), 9)
        col_base_y = np.tile(col_base + 3, ny - 1) + np.repeat(3 * np.arange(ny - 1), 9)
        te_od_row = np.tile(row_base_y, nx) + np.repeat(nn2 * np.arange(nx), 9 * (ny - 1))
        le_od_col = np.tile(col_base_y, nx)
        te_od_col = le_od_col + 3 * ny * (nx - 1)
        rows = np.concatenate([dg_row, le_dg_row, te_dg_row, te_od_row, te_od_row])
        cols = np.concatenate([dg_col, le_dg_col, te_dg_col, le_od_col, te_od_col])
    else:
        n_sym = (ny - 1) // 2
        row_base_y1 = np.tile(row_base, n_sym) + np.repeat(3 * np.arange(n_sym), 9)
        col_base_y1 = np.tile(col_base + 3, n_sym) + np.repeat(3 * np.arange(n_sym), 9)
        row_base_y2 = row_base_y1 + 3 * n_sym + 3
        col_base_y2 = col_base_y1 + 3 * n_sym - 3
        te_od_row1 = np.tile(row_base_y1, nx) + np.repeat(nn2 * np.arange(nx), 9 * n_sym)
        le_od_col1 = np.tile(col_base_y1, nx)
        te_od_col1 = le_od_col1 + 3 * ny * (nx - 1)
        te_od_row2 = np.tile(row_base_y2, nx) + np.repeat(nn2 * np.arange(nx), 9 * n_sym)
        le_od_col2 = np.tile(col_base_y2, nx)
        te_od_col2 = le_od_col2 + 3 * ny * (nx - 1)
        rows = np.concatenate([dg_row, le_dg_row, te_dg_row, te_od_row1, te_od_row2, te_od_row1, te_od_row2])
        cols = np.concatenate([dg_col, le_dg_col, te_dg_col, le_od_col1, le_od_col2, te_od_col1, te_od_col2])

    return (twist_rows, twist_cols), (rows, cols)


class Rotate(om.ExplicitComponent):
    """
    Manipulate the mesh by rotating each spanwise station by its twist angle about the reference axis.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    twist[ny] : numpy array
        1-D array of rotation angles about y-axis for each wing slice in degrees.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the twisted aerodynamic surface.
    """

    val: FloatOrNpArray = Field(description="Initial value for twist.")
    mesh_shape: MeshShape = Field(description="Mesh shape (nx, ny, 3).")
    symmetry: bool = Field(default=False, description="True if surface is reflected about y=0 plane.")
    rotate_x: bool = Field(
        default=True,
        description="True to always apply the twist perpendicular to the wing (say, for a winglet).",
    )
    ref_axis_pos: float = Field(default=0.25, description="Fraction of the chord to use as the reference axis.")

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        mesh_shape = tuple(f.mesh_shape)
        (t_rows, t_cols), (m_rows, m_cols) = _rotate_pattern(mesh_shape, f.symmetry)
        data["inputs"] = {
            "twist": om.InputVar(val=np.asarray(f.val, dtype=float), units="deg"),
            "in_mesh": om.InputVar(val=np.ones(mesh_shape), units="m"),
        }
        data["outputs"] = {"mesh": om.OutputVar(val=np.ones(mesh_shape), units="m")}
        data["partials"] = [
            om.PartialsSpec(of="mesh", wrt="twist", rows=t_rows, cols=t_cols),
            om.PartialsSpec(of="mesh", wrt="in_mesh", rows=m_rows, cols=m_cols),
        ]
        return data

    @cached_property
    def _n_in_mesh(self) -> int:
        """Number of declared nonzeros in d mesh / d in_mesh."""
        return len(_rotate_pattern(tuple(self.mesh_shape), self.symmetry)[1][0])

    def _theta_x(self, ref_axis, ny):
        """x-axis rotation angle distribution from the spanwise z displacements along the reference axis."""
        if not self.rotate_x:
            return 0.0
        if self.symmetry:
            dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
            dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
            theta_x = np.arctan(dz_qc / dy_qc)
            # Append 0 so that root is not rotated
            return np.append(theta_x, 0.0)
        root_index = int((ny - 1) / 2)
        dz_qc_left = ref_axis[:root_index, 2] - ref_axis[1 : root_index + 1, 2]
        dy_qc_left = ref_axis[:root_index, 1] - ref_axis[1 : root_index + 1, 1]
        theta_x_left = np.arctan(dz_qc_left / dy_qc_left)
        dz_qc_right = ref_axis[root_index + 1 :, 2] - ref_axis[root_index:-1, 2]
        dy_qc_right = ref_axis[root_index + 1 :, 1] - ref_axis[root_index:-1, 1]
        theta_x_right = np.arctan(dz_qc_right / dy_qc_right)
        return np.concatenate((theta_x_left, np.zeros(1), theta_x_right))

    @staticmethod
    def _rotation_mats(rad_theta_x, rad_theta_y, ny):
        """Rx(theta_x) Ry(theta_y) at each spanwise station, shape (ny, 3, 3)."""
        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))
        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)
        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx * cos_rty
        return mats

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]
        ref_axis = _ref_axis(mesh, self.ref_axis_pos)
        _, ny, _ = mesh.shape

        rad_theta_x = self._theta_x(ref_axis, ny)
        rad_theta_y = theta_y * np.pi / 180.0
        mats = self._rotation_mats(rad_theta_x, rad_theta_y, ny)

        # i - spanwise station, m - chordwise station, k - rotated vector, j - input vector
        outputs["mesh"] = np.einsum("ikj, mij -> mik", mats, mesh - ref_axis) + ref_axis

    def compute_partials(self, inputs, partials):
        symmetry = self.symmetry
        rotate_x = self.rotate_x
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]
        rap = self.ref_axis_pos
        ref_axis = _ref_axis(mesh, rap)
        nx, ny, _ = mesh.shape

        if rotate_x:
            if symmetry:
                dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
                dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
                fact = 1.0 / (1.0 + (dz_qc / dy_qc) ** 2)

                # Derivative of theta_x wrt the y and z components of the ref_axis
                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:-1, 1] = -dz_qc * fact / dy_qc**2
                dthx_dq[:-1, 2] = fact / dy_qc
            else:
                root_index = int((ny - 1) / 2)
                dz_qc_left = ref_axis[:root_index, 2] - ref_axis[1 : root_index + 1, 2]
                dy_qc_left = ref_axis[:root_index, 1] - ref_axis[1 : root_index + 1, 1]
                dz_qc_right = ref_axis[root_index + 1 :, 2] - ref_axis[root_index:-1, 2]
                dy_qc_right = ref_axis[root_index + 1 :, 1] - ref_axis[root_index:-1, 1]
                fact_left = 1.0 / (1.0 + (dz_qc_left / dy_qc_left) ** 2)
                fact_right = 1.0 / (1.0 + (dz_qc_right / dy_qc_right) ** 2)

                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:root_index, 1] = -dz_qc_left * fact_left / dy_qc_left**2
                dthx_dq[root_index + 1 :, 1] = -dz_qc_right * fact_right / dy_qc_right**2
                dthx_dq[:root_index, 2] = fact_left / dy_qc_left
                dthx_dq[root_index + 1 :, 2] = fact_right / dy_qc_right

        rad_theta_x = self._theta_x(ref_axis, ny)
        deg2rad = np.pi / 180.0
        rad_theta_y = theta_y * deg2rad
        mats = self._rotation_mats(rad_theta_x, rad_theta_y, ny)

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        # Derivative of the rotation matrices wrt twist
        dmats_dthy = np.zeros((ny, 3, 3))
        dmats_dthy[:, 0, 0] = -sin_rty * deg2rad
        dmats_dthy[:, 0, 2] = cos_rty * deg2rad
        dmats_dthy[:, 1, 0] = sin_rtx * cos_rty * deg2rad
        dmats_dthy[:, 1, 2] = sin_rtx * sin_rty * deg2rad
        dmats_dthy[:, 2, 0] = -cos_rtx * cos_rty * deg2rad
        dmats_dthy[:, 2, 2] = -cos_rtx * sin_rty * deg2rad

        d_dthetay = np.einsum("ikj, mij -> mik", dmats_dthy, mesh - ref_axis)
        partials["mesh", "twist"] = d_dthetay.flatten()

        # d mesh / d in_mesh, assembled locally (entries not written below stay zero, as in OM3)
        d = np.zeros(self._n_in_mesh)

        # Rotation matrices on the diagonal, tiled for every chordwise row
        nn = nx * ny * 9
        d[:nn] = np.tile(mats.flatten(), nx)

        # Reference axis direct contribution: (I - R) at each spanwise station
        eye = np.tile(np.eye(3).flatten(), ny).reshape(ny, 3, 3)
        d_qch = (eye - mats).flatten()
        nqc = ny * 9
        d[:nqc] += (1 - rap) * d_qch
        d[nn - nqc : nn] += rap * d_qch

        if rotate_x:
            # LE/TE terms that exist because theta_x depends on the reference axis geometry
            dmats_dthx = np.zeros((ny, 3, 3))
            dmats_dthx[:, 1, 0] = cos_rtx * sin_rty
            dmats_dthx[:, 1, 1] = -sin_rtx
            dmats_dthx[:, 1, 2] = -cos_rtx * cos_rty
            dmats_dthx[:, 2, 0] = sin_rtx * sin_rty
            dmats_dthx[:, 2, 1] = cos_rtx
            dmats_dthx[:, 2, 2] = -sin_rtx * cos_rty

            d_dthetax = np.einsum("ikj, mij -> mik", dmats_dthx, mesh - ref_axis)
            d_dq = np.einsum("ijk, jm -> ijkm", d_dthetax, dthx_dq)
            d_dq_flat = d_dq.flatten()

            del_n = nn - 9 * ny
            nn2 = nn + del_n
            nn3 = nn2 + del_n

            # LE contribution excludes LE partials; TE contribution excludes TE partials
            d[nn:nn2] = (1 - rap) * d_dq_flat[-del_n:]
            d[nn2:nn3] = rap * d_dq_flat[:del_n]

            # Contributions back to the main diagonal (a single LE or TE row)
            del_n = 9 * ny
            d[:nqc] += (1 - rap) * d_dq_flat[:del_n]
            d[nn - nqc : nn] += rap * d_dq_flat[-del_n:]

            # Position of the reference axis itself
            d_qch_od = np.tile(d_qch.flatten(), nx - 1)
            d[nn:nn2] += (1 - rap) * d_qch_od
            d[nn2:nn3] += rap * d_qch_od

            # Off-off-diagonal: sensitivity to spanwise station positions relative to the root
            if symmetry:
                d_dq_flat = d_dq[:, :-1, :, :].flatten()
                del_n = nn - 9 * nx
                nn4 = nn3 + del_n
                d[nn3:nn4] = -(1 - rap) * d_dq_flat
                nn5 = nn4 + del_n
                d[nn4:nn5] = -rap * d_dq_flat
            else:
                d_dq_flat1 = d_dq[:, :root_index, :, :].flatten()
                d_dq_flat2 = d_dq[:, root_index + 1 :, :, :].flatten()
                del_n = nx * root_index * 9
                nn4 = nn3 + del_n
                d[nn3:nn4] = -(1 - rap) * d_dq_flat1
                nn5 = nn4 + del_n
                d[nn4:nn5] = -(1 - rap) * d_dq_flat2
                nn6 = nn5 + del_n
                d[nn5:nn6] = -rap * d_dq_flat1
                nn7 = nn6 + del_n
                d[nn6:nn7] = -rap * d_dq_flat2

        partials["mesh", "in_mesh"] = d
