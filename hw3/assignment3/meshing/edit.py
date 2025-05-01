from typing import Optional, Tuple
from dataclasses import dataclass
from . import Halfedge, Edge, Vertex, Face, Topology, Mesh
import numpy as np


"""
TODO complete these functions
P5 -- LaplacianSmoothing.apply
P6 -- prepare_collapse, do_collapse
    (+ add your own fields to the struct CollapsePrep as needed)
Extra credit -- prepare_collapse_with_link
    If you have P6 working, this part will look very similar, just an extra
    check to add to prepare_collapse.
"""


class MeshEdit:
    """
    Abstract interface for a mesh edit. The edit is prepared upon init
    (creating/storing info about the edit before actually executing it) then, if
    determined to be doable, applied with apply().
    """

    def __init__(self):
        pass

    def apply(self):
        pass


class LaplacianSmoothing(MeshEdit):
    def __init__(self, mesh: Mesh, n_iter: int):
        self.mesh = mesh
        self.n_iter = n_iter

    def apply(self):
        for i in range(self.n_iter):
            new_vertice_positions = []

            # Assuming that the vertices are arranged from 0th index to last
            # in self.topology.vertices.values().
            for v in self.mesh.topology.vertices.values():
                neighbors = v.adjacentVertices()
                neighbor_positions = np.array([self.mesh.get_3d_pos(n) for n in neighbors])
                avg_x = np.mean(neighbor_positions[:,0])
                avg_y = np.mean(neighbor_positions[:,1])
                avg_z = np.mean(neighbor_positions[:,2])

                p = (avg_x, avg_y, avg_z)
                new_vertice_positions.append(p)
            
            # Updating the vertices
            self.mesh.vertices = np.array(new_vertice_positions)

# TODO: P6 -- add fields to this to use in your collapse
@dataclass
class CollapsePrep:
    """
    Basically a struct that packages information about one collapse operation.
    TODO add your own fields needed to execute a prepared collapse operation,
    namely, references to relevant halfedges and other Primitive objects,
    and any extra information you may need.
    """

    pass


class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        return do_collapse(self.prep, self.mesh, self.e_id)


# TODO: P6 -- complete this
def prepare_collapse(mesh: Mesh, e_id: int) -> CollapsePrep:
    """
    We encourage you to split up the collapse task into a prepare_collapse and
    an do_collapse stage because it makes it easier to see two parts: 1)
    traversing the halfedge structure to gather up the relevant Primitives
    without making changes, and 2) making changes, including reassigning the
    linkages in the halfedge structure and deleting the Primitive objects
    belonging to the vertices, faces, and edges removed in the operation.

    To delete primitives, for instance, a halfedge with index halfedge_id, use
        del mesh.topology.halfedges[halfedge_id]
    and similarly for other primitive types.

    (For the Extra Credit, having an edge collapse preparation stage also means
    you can perform a check at preparation and determine if the collapse is
    doable at all.)
    """
    topology = mesh.topology
    e = topology.edges[e_id]
    # TODO write your code here and replace this raise and return
    raise NotImplementedError("TODO (P6)")


# TODO: P6 -- complete this
def do_collapse(prep: CollapsePrep, mesh: Mesh, e_id: int):
    """
    TODO complete this function.
    This should modify the mesh's topology and vertices coords inplace.
    (You should not need to create any new Primitives.)
    """
    raise NotImplementedError("TODO (P6)")


class EdgeCollapseWithLink(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.prep = prepare_collapse_with_link_cond(self.mesh, self.e_id)
        # ^ this may be None/falsy, in which case this is not doable

    def apply(self):
        if not self.prep:
            print(f"Collapse is not doable, does not satisfy link condition")
            return
        # notice that you can just reuse your P6 collapse execution stage.
        return do_collapse(self.prep, self.mesh, self.e_id)


# TODO: Extra credit -- complete this
def prepare_collapse_with_link_cond(mesh: Mesh, e_id: int) -> Optional[CollapsePrep]:
    """
    You should perform the link condition check in addition to your
    prepare_collapse and if it is not met, you can just return None early.
    """
    topology = mesh.topology
    e = topology.edges[e_id]
    # TODO write your code here and replace this raise and return
    raise NotImplementedError("TODO (EC)")
