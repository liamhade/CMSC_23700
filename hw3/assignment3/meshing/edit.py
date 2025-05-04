from typing import Optional, Tuple, List
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

def average_3d_point(points: np.array) -> np.array:
    """
    Given an arbitrary number of 3D points, this function
    finds the average point among all of them (i.e. the centroid).
    """
    avg_x = np.mean(points[:, 0])
    avg_y = np.mean(points[:, 1])
    avg_z = np.mean(points[:, 2])
    return (avg_x, avg_y, avg_z)

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
                p = average_3d_point(neighbor_positions)
                new_vertice_positions.append(p)
            
            # Updating the vertices
            self.mesh.vertices = np.array(new_vertice_positions)

@dataclass
class CollapsePrep:
    """
    Basically a struct that packages information about one collapse operation.
    TODO add your own fields needed to execute a prepared collapse operation,
    namely, references to relevant halfedges and other Primitive objects,
    and any extra information you may need.
    """
    '''
    Dictionary that tells you which edge was collapsed /
    subsumed into another edge. We want this to be a class variable
    so that it gets updated and maintained across CollapsePrep instances.

    The structure of this variable is of the type:
        * key   : collapsed / removed edge_id.
        * value : edge_id of the edge that overwrote it.
    '''
    collapsed_edges = {} 

    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh

        # We might have overwritten this edge earlier by some other edge,
        # but we still want to be able to collapse it.
        # Using a loop incase there are any transitive
        # effects we need to capture.
        while e_id in self.collapsed_edges:
            e_id = self.collapsed_edges[e_id]

        self.e = self.mesh.topology.edges[e_id]

        # Grabbing the two vertices that are going to be deleted
        collapsed_vertices = self.e.two_vertices()
        # We could have picked either one of the two vertices in our edge to delete.
        self.vertex_to_delete = collapsed_vertices[0]
        # The other vertex will be reassigned to have a position equal to our midpoint.
        self.remaining_vertex = collapsed_vertices[1]

        # Converting the vertices into 3D mesh positions
        v1, v2 = collapsed_vertices # Unpacking the vertices tuple 
        p1 = self.mesh.get_3d_pos(v1)
        p2 = self.mesh.get_3d_pos(v2)
        # Midpoint of the edge we're collapsing.
        self.midpoint = average_3d_point(np.array([p1, p2]))

        # Used for reassigning mesh.vertices
        self.vertex_reassignment = {self.remaining_vertex: self.midpoint}

        # Faces that are adjacent to both edge vertices must also be adjacent to the edge.
        self.faces_to_delete = list(set.intersection(set(v1.adjacentFaces()), set(v2.adjacentFaces())))

        # List of all the halfedges we'll end up deleting
        self.halfedges_to_delete = [he for f in self.faces_to_delete for he in f.adjacentHalfedges()]

        self.edges_to_delete = [self.e]

        # Because of our deletions, certain halfedges that once had 
        # an edge and a halfedge will be stranded.
        self.halfedge_reassignments = {}

        # The halfedges that we delete are only those inside of the faces we're going to delete.
        # This information is useful when we want to figure out how to reassign some "floating" edges.
        for he in self.halfedges_to_delete:
            if self.only_tail_on_collapsed_vertex(he, collapsed_vertices):
                # The twin of this halfedge will be stranded
                self.edges_to_delete.append(he.edge)
                
                # Reassigning the newly stranded halfedge
                stranded_halfedge = he.twin
                new_twin = he.next.twin
                new_edge = new_twin.edge

                # If we try to delete this edge again then
                # we'll map it to this new edge.
                self.collapsed_edges[he.edge.index] = new_edge.index

                self.halfedge_reassignments[stranded_halfedge] = {"new_twin"   : new_twin, 
                                                                  "new_edge"   : new_edge, 
                                                                  "new_vertex" : stranded_halfedge.vertex}

        # Looking at the halfedges around our deleted vertex
        # that will remain after being deleted
        for he in self.vertex_to_delete.adjacentHalfedges():
            if he in self.halfedges_to_delete:
                # We don't need to reassign this halfedge since
                # it's going to be deleted.
                continue
            else:
                '''
                We know that this halfedge isn't already in our halfedge_reassignments
                since we only looked at those halfedges that were pointing towards
                one of our deleted edge vertices -- that is, we only looked at the twins of the halfedges
                inside of our deleted faces whose tail was on a deleted vertex.
                '''
                self.halfedge_reassignments[he] = {"new_twin"   : he.twin,
                                                   "new_edge"   : he.edge,
                                                   "new_vertex" : self.remaining_vertex}

    def only_tail_on_collapsed_vertex(self, he: Halfedge, collapsed_vertices: List[Vertex]) -> bool:
        return (he.vertex in collapsed_vertices) and (he.tip_vertex() not in collapsed_vertices)
                    
class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        return do_collapse(self.prep, self.mesh, self.e_id)


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
    return CollapsePrep(mesh, e_id)


def do_collapse(prep: CollapsePrep, mesh: Mesh, e_id: int):
    '''
    Deletion
    '''

    for f in set(prep.faces_to_delete):
        del mesh.topology.faces[f.index]

    # Making it a set since we don't want to delete any edges twice.
    for e in set(prep.edges_to_delete):
        del mesh.topology.edges[e.index]

    for d_he in set(prep.halfedges_to_delete):
        del mesh.topology.halfedges[d_he.index]

    del mesh.topology.vertices[prep.vertex_to_delete.index]

    '''
    Reassignment
    '''
    for he, assignment in prep.halfedge_reassignments.items():
        he.twin   = assignment["new_twin"]
        he.edge   = assignment["new_edge"]
        he.vertex = assignment["new_vertex"]
    
    mesh.vertices[prep.remaining_vertex.index] = prep.midpoint


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
