from typing import Any

import numpy as np
from pydantic import Field, model_validator

import om4.api as om
from om4.utils.types import FloatOrNpArray

from openaerostruct.utils.om4_utils import VarDecl, field_values
from openaerostruct.utils.surface import Surface

default_vec = np.ones(3)


def get_section_edge_left(mesh, v=default_vec, edge_cur=0, edges_all_constraints=default_vec):
    """
    Function that gets the coordinates of the leading and trailing edge points of the left edge of a section. The output can be masked to only retreive the x,y, or z coordinate.
    The function also returns the row and column vectors of non zero entries in the edge coordinate jacobian.

    Parameters
    ----------
    mesh : numpy array
        OAS mesh of a given section
    v : numpy array[3]
        Numpy array of length three populuted with ones and zeros used to mask the output so that only the specified of the x, y, and z coordinates are returned.
    edge_cur : int
        Integer indicating which unique intersection of edges this particular edge is associated with. Required to return the correct jacobian sparsity pattern.
    edges_all_constraints: list
        See dim_constr in the GeomMultiJoin component. This array needs to passed into this function from the component in order to return the correct jacobian sparsity pattern.

    Returns
    -------
    edge points : numpy array
        Array of points corresponding to the leading and trailing edges of the left section edge. Masked according to input.
    rows : numpy array
        Array of the rows of the non-zero jacobian entries
    cols : numpy array
        Array of the columns of the non-zero jacobian entries

    """
    nx = mesh.shape[0]
    le_index = 0
    te_index = np.ravel_multi_index((nx - 1, 0, 0), mesh.shape)
    mask = np.array(v, dtype="bool")

    rows = np.arange(0, 2 * np.sum(v)) + 2 * int(np.sum(edges_all_constraints[:edge_cur]))
    cols = np.concatenate([np.arange(le_index, le_index + 3)[mask], np.arange(te_index, te_index + 3)[mask]])

    return mesh[[0, -1], 0][:, np.arange(0, 3)[mask]], rows, cols


def get_section_edge_right(mesh, v=default_vec, edge_cur=0, edges_all_constraints=default_vec):
    """
    Function that gets the coordinates of the leading and trailing edge points of the right edge of a section. The output can be masked to only retreive the x,y, or z coordinate.
    The function also returns the row and column vectors of non zero entries in the edge coordinate jacobian.

    Parameters
    ----------
    mesh : numpy array
        OAS mesh of a given section
    v : numpy array[3]
        Numpy array of length three populuted with ones and zeros used to mask the output so that only the specified of the x, y, and z coordinates are returned.
    edge_cur : int
        Integer indicating which unique intersection of edges this particular edge is associated with. Required to return the correct jacobian sparsity pattern.
    edges_all_constraints: list
        See dim_constr in the GeomMultiJoin component. This array needs to passed into this function from the component in order to return the correct jacobian sparsity pattern.

    Returns
    -------
    edge points : numpy array
        Array of points corresponding to the leading and trailing edges of the right section edge. Masked according to input.
    rows : numpy array
        Array of the rows of the non-zero jacobian entries
    cols : numpy array
        Array of the columns of the non-zero jacobian entries
    """

    nx = mesh.shape[0]
    ny = mesh.shape[1]
    le_index = np.ravel_multi_index((0, ny - 1, 0), mesh.shape)
    te_index = np.ravel_multi_index((nx - 1, ny - 1, 0), mesh.shape)
    mask = np.array(v, dtype="bool")

    rows = np.arange(0, 2 * np.sum(v)) + 2 * int(np.sum(edges_all_constraints[:edge_cur]))
    cols = np.concatenate([np.arange(le_index, le_index + 3)[mask], np.arange(te_index, te_index + 3)[mask]])

    return mesh[[0, -1], -1][:, np.arange(0, 3)[mask]], rows, cols


def _resolve_dim_constr(num_sections, dim_constr):
    """Default to the distance along the x-axis for each intersecting edge unless one entry per edge is given."""
    if len(dim_constr) != num_sections - 1:
        return [np.array([1, 0, 0]) for _ in range(num_sections - 1)]
    return dim_constr


class GeomMultiJoin(om.ExplicitComponent):
    """
    OpenMDAO component that outputs the distance between the leading and trailing edge corners of each section corresponding to each shared edge.
    Users can specify along which axes the distance should be computed([x y z]).

    Parameters
    ----------
    sections : list
        List of section OpenAeroStruct surface dictionaries
    dim_constr : list
        A list of vectors of length three corresponding to each edge. Entries correspond to the dimension([x,y,z]) the user wishes to constraint should be set to 1. Remaining entries should be zero.

    Returns
    -------
    section_separation[2*count_nonzero(concatenate(dim_constr)] : numpy array
        Numpy array of distances along each axis between corresponding leading and trailing edge points between subsequent sections.
    """

    sections: list[Surface]
    dim_constr: list[FloatOrNpArray] = Field(default_factory=lambda: [np.ones(3)])

    @model_validator(mode="before")
    @classmethod
    def _build_vars(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        f = field_values(cls, data)
        sections = f.sections
        dim_constr = _resolve_dim_constr(len(sections), f.dim_constr)
        data["dim_constr"] = dim_constr
        d = VarDecl()

        # Compute size of and add output
        constr_size = 2 * np.count_nonzero(np.concatenate(dim_constr))
        d.add_output("section_separation", val=np.zeros(constr_size))

        # Generate the Jacobian of the edge seperation distance with respect to the section mesh.
        # Jacobian is just ones, zeros, and negative ones so we can declare here. However the sparsity pattern is complicated and requires two helper functions and the loop below.

        # Counter used to track the current unique edge interection between section being processed.
        edge_cur = 0

        for i_sec, section in enumerate(sections):
            mesh = section["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = section["name"]

            # Add the input
            mesh_name = "{}_join_mesh".format(name)
            d.add_input(mesh_name, shape=(nx, ny, 3), units="m")

            # Get the sparsity patterns for each section. First and last sections only have one edge intersection.
            if i_sec == 0:
                rows, cols = get_section_edge_right(mesh, dim_constr[i_sec], edge_cur, dim_constr)[1:]
                vals = -1 * np.ones_like(rows)
            elif i_sec < len(sections) - 1:
                rows1, cols1 = get_section_edge_left(mesh, dim_constr[i_sec - 1], edge_cur, dim_constr)[1:]
                vals1 = np.ones_like(rows1)

                edge_cur += 1
                rows2, cols2 = get_section_edge_right(mesh, dim_constr[i_sec], edge_cur, dim_constr)[1:]
                vals2 = -1 * np.ones_like(rows2)

                rows = np.concatenate([rows1, rows2])
                cols = np.concatenate([cols1, cols2])
                vals = np.concatenate([vals1, vals2])
            else:
                rows, cols = get_section_edge_left(mesh, dim_constr[i_sec - 1], edge_cur, dim_constr)[1:]
                vals = np.ones_like(rows)

            # Declare partials for the current section
            d.declare_partials("section_separation", mesh_name, rows=rows, cols=cols, val=vals)
        return d.into(data)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Compute the distances between the corresponding leading and trailing edges along the edge interection between each section
        sections = self.sections
        dim_constr = self.dim_constr
        edges = []
        edge_constraints = []
        for i_sec, section in enumerate(sections):
            name = section["name"]
            mesh_name = "{}_join_mesh".format(name)

            if i_sec == 0:
                edges.append(get_section_edge_right(inputs[mesh_name], dim_constr[i_sec])[0])
            elif i_sec < len(sections) - 1:
                edges.append(get_section_edge_left(inputs[mesh_name], dim_constr[i_sec - 1])[0])
                edges.append(get_section_edge_right(inputs[mesh_name], dim_constr[i_sec])[0])
            else:
                edges.append(get_section_edge_left(inputs[mesh_name], dim_constr[i_sec - 1])[0])

        for i in range(len(sections) - 1):
            edge_constraints.append((edges[2 * i + 1] - edges[2 * i]).flatten())

        outputs["section_separation"] = np.array(edge_constraints).flatten()
