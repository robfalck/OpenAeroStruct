"""Utilities for splitting a `MultiSectionSurface` into sections and stitching them back together."""

import copy

import numpy as np

import om4.api as om
from om4.core.group import Connection

from openaerostruct.meshing.section_mesh_generator import generate_mesh
from openaerostruct.utils.surface import MultiSectionSurface, Surface

# List of supported keys for multi-section wings.
# NOTE: make sure this is consistent with the documentation's surface dict page.
_SECTION_KEYS = [
    # Essential Info
    "symmetry",
    "S_ref_type",
    "ref_axis_pos",
    # wing definition
    "span",
    "taper",
    "sweep",
    "dihedral",
    "twist_cp",
    "chord_cp",
    "xshear_cp",
    "yshear_cp",
    "zshear_cp",
    # aerodynamics
    "CL0",
    "CD0",
    "with_viscous",
    "with_wave",
    "groundplane",
    "k_lam",
    "t_over_c_cp",
    "c_max_t",
]


def build_section_dicts(surface: MultiSectionSurface | dict) -> list[Surface]:
    """
    Split a multi-section surface into one ``Surface`` per section.

    Ported mechanically from OM3's dict-based version (``openaerostruct.geometry.utils``, pre-
    om4).  ``generate_mesh`` mutates its input dict in place (coercing scalars to per-section
    lists), so it is called on a plain dict copy of ``surface`` rather than on the read-only
    ``MultiSectionSurface`` model.  A plain dict is accepted directly too (coerced to
    ``MultiSectionSurface`` first), so callers may keep mutating a dict before calling this,
    as several existing tests already do.

    Parameters
    ----------
    surface : MultiSectionSurface | dict
        Multi-section surface.

    Returns
    -------
    list[Surface]
        One surface per section, named per ``surface.sec_name``.
    """
    if not isinstance(surface, MultiSectionSurface):
        surface = MultiSectionSurface.model_validate(surface)

    num_sections = surface.num_sections

    if surface.meshes == "gen-meshes":
        if len(surface.ny) != num_sections:
            raise ValueError("Number of spanwise points needs to be provided for each section")
        if len(surface.taper) != num_sections:
            raise ValueError("Taper needs to be provided for each section")
        if len(surface.span) != num_sections:
            raise ValueError("Span needs to be provided for each section")
        if len(surface.sweep) != num_sections:
            raise ValueError("Sweep needs to be provided for each section")

        _, sec_meshes = generate_mesh(surface.model_dump(exclude_none=True))
    else:
        if len(surface.meshes) != num_sections:
            raise ValueError("A mesh needs to be provided for each section.")
        sec_meshes = surface.meshes

    if len(surface.sec_name) != num_sections:
        raise ValueError("A name needs to be provided for each section.")

    # OM3 iterated `set(surface).intersection(target_keys)`; `MultiSectionSurface` has no
    # `__iter__` (only the read-only mapping protocol), so iterate the key list instead and
    # skip keys the surface doesn't define -- same resulting set, order doesn't matter here.
    surface_sections = []
    for i in range(num_sections):
        section = {}
        for k in _SECTION_KEYS:
            if k not in surface:
                continue
            val = surface[k]
            if isinstance(val, list):
                # Reset taper and sweep so that OAS doesn't apply the transformations again.
                if k == "taper":
                    section[k] = 1.0
                elif k == "sweep":
                    section[k] = 0.0
                elif k == "span":
                    section[k] = 2.0 * val[i] if surface.symmetry else val[i]
                else:
                    section[k] = val[i]
            else:
                section[k] = val
        section["mesh"] = sec_meshes[i]
        section["name"] = surface.sec_name[i]
        surface_sections.append(Surface(**section))
    return surface_sections


# Fields on `MultiSectionSurface` that are per-section lists (or otherwise unique to it) and have
# no corresponding scalar field on `Surface`; dropped by `scalar_overlay` below.
_MULTISEC_ONLY_FIELDS = {
    "taper",
    "span",
    "sweep",
    "dihedral",
    "twist_cp",
    "chord_cp",
    "xshear_cp",
    "yshear_cp",
    "zshear_cp",
    "t_over_c_cp",
    "num_sections",
    "sec_name",
    "meshes",
    "ny",
}


def scalar_overlay(surface: MultiSectionSurface | dict) -> dict:
    """
    Return `surface`'s cross-section-shared (non-per-section) fields as a plain dict.

    Suitable for feeding `Surface`-typed subsystems (`VLMGeometry`/`VLMFunctionals`/`TubeGroup`/
    `WingboxGroup`/`SpatialBeamSetup`/...) that only read scalar, shared surface properties plus
    `mesh`/`t_over_c` -- the caller is expected to have already written the unified mesh (and
    t_over_c, if active) into `surface["mesh"]` via `unify_mesh` before calling this, mirroring
    `Geometry._resolve_gen_mesh`'s "gen-mesh" convention.
    """
    if isinstance(surface, MultiSectionSurface):
        return surface.model_dump(exclude=_MULTISEC_ONLY_FIELDS, exclude_none=True)
    return {k: v for k, v in surface.items() if k not in _MULTISEC_ONLY_FIELDS}


def unify_mesh(sections: list[Surface], shift_uni_mesh: bool = True) -> np.ndarray:
    """
    Produce a unified mesh from all the individual wing section meshes.

    Parameters
    ----------
    sections : list[Surface]
        List of section surfaces, as returned by ``build_section_dicts``.
    shift_uni_mesh : bool
        If True, shift sections so that their leading edges are coincident, to keep sections
        from separating or intersecting during scalar span or sweep operations without the use
        of the constraint component (``GeomMultiJoin``).

    Returns
    -------
    np.ndarray
        Unified surface mesh in OAS format.
    """
    for i_sec in np.arange(0, len(sections) - 1):
        mesh = sections[i_sec]["mesh"]

        if i_sec == 0:
            uni_mesh = copy.deepcopy(mesh[:, :-1, :])
        else:
            if shift_uni_mesh:
                # Translate/shift uni_mesh (outer sections) to align the leading edge at the
                # unification boundary.
                last_mesh = sections[i_sec - 1]["mesh"]
                uni_mesh = uni_mesh - last_mesh[0, -1, :] + mesh[0, 0, :]

            uni_mesh = np.concatenate([uni_mesh, mesh[:, :-1, :]], axis=1)

    # Stitch the results into a singular mesh.
    mesh = sections[len(sections) - 1]["mesh"]
    if len(sections) == 1:
        uni_mesh = copy.deepcopy(mesh)
    else:
        uni_mesh = np.concatenate([uni_mesh, mesh], axis=1)

    return uni_mesh


def multi_spline_join_kwargs(
    out_name: str,
    section_surfaces: list[Surface],
    sec_cp: list[np.ndarray],
    geom_name: str,
    units: str | None = None,
) -> dict:
    """
    Declarative replacement for OM3's ``build_multi_spline``/``connect_multi_spline``.

    Joins each section's B-spline control points "by construction": adjacent sections share
    their edge control point, so the free control points across all sections form one shorter
    vector.  OM3 built an ``IndepVarComp`` and connected it to each section's spline input after
    the ``Problem`` existed (``prob.model.connect(..., src_indices=...)``); om4 builds everything
    up front and has no ``IndepVarComp``, so here a passthrough ``ExecComp`` (the same holder
    idiom used by ``spline_or_broadcast``) holds the free vector on a promoted input -- the
    unconnected DV target, same convention every other OAS DV already uses -- and its output is
    sliced into each section's control-point input via explicit ``Connection``s.

    Parameters
    ----------
    out_name : str
        Name of the control-point input each section's ``Geometry`` promotes (e.g. "twist_cp").
    section_surfaces : list[Surface]
        Per-section surfaces, as returned by ``build_section_dicts``.
    sec_cp : list[np.ndarray]
        Control-point array for each section.  A length-1 section has no edge point of its own
        to share, so it cannot enforce continuity with its neighbor by construction; such
        sections are reported in ``bind_inds`` for the caller to constrain manually.
    geom_name : str
        Name of the subsystem the per-section ``Geometry`` groups are nested under (the
        ``MultiSecGeometry``/``MultiSecAerostructGeometry`` subsystem key in the caller's own
        top-level group).
    units : str, optional
        Units of the control points.

    Returns
    -------
    dict
        ``{"subsystems": {...}, "connections": [...], "bind_inds": [...]}`` for the caller to
        merge into its own top-level ``om.Group``.
    """
    if len(sec_cp) != len(section_surfaces):
        raise ValueError("Target sections need to match with control points!")

    single_sections = len([cp for cp in sec_cp if len(cp) == 1])
    n_free = len(np.concatenate(sec_cp)) - (len(sec_cp) - 1 - single_sections)

    holder_name = f"{out_name}_spline"
    holder_in = f"_{holder_name}_1d"
    holder_out = f"_{holder_name}_out"
    holder = om.ExecComp(
        exprs=[f"{holder_out} = {holder_in}"],
        inputs={holder_in: {"units": units, "val": np.ones(n_free)}},
        outputs={holder_out: {"units": units, "shape": (n_free,)}},
        promotes_inputs=[(holder_in, holder_name)],
    )

    connections = []
    bind_inds = []
    acc = 0
    for section, cp in zip(section_surfaces, sec_cp):
        point_count = len(cp)
        src_inds = np.arange(acc, acc + point_count)
        acc += point_count - 1
        if point_count == 1:
            acc += 1
            bind_inds.append(acc)
        connections.append(
            Connection(
                src=f"{holder_name}.{holder_out}",
                tgt=f"{geom_name}.{section['name']}.{out_name}",
                src_indices=src_inds,
            )
        )

    return {"subsystems": {holder_name: holder}, "connections": connections, "bind_inds": bind_inds}
