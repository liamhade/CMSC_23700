import numpy as np
from collections import defaultdict
from typing import Dict, Callable, TypeVar, Tuple
from .primitive import Halfedge, Vertex, Edge, Face

"""
TODO complete these functions
P1 -- Topology.build
P3 -- Topology.hasNonManifoldVertices, Topology.hasNonManifoldEdges

NOTE that Topology.thorough_check won't work all the way through until you finish
both P1 and Vertex.adjacentHalfedges, Face.adjacentHalfedges functions in P2 in primitive.py
"""

ElemT = TypeVar("ElemT", Halfedge, Vertex, Edge, Face)


class ElemCollection(Dict[int, ElemT]):
    """
    This dict wrapper keeps track of the number of uniquely allocated elements so that each
    element has an unambiguous index/key (among objects of the same type) in the lifetime of
    the mesh (even after edits)
    """

    def __init__(self, constructor: Callable[[], ElemT]):
        super().__init__()
        self.cons_f = constructor
        self.n_allocations = 0

    def allocate(self) -> ElemT:
        elem = self.cons_f()
        i = self.n_allocations
        elem.index = i
        self[i] = elem
        self.n_allocations += 1
        return elem

    def fill_vacant(self, elem_id: int):
        """an element was previously deleted, and is now re-inserted"""
        assert elem_id not in self
        elem = self.cons_f()
        elem.index = elem_id
        self[elem_id] = elem
        return elem

    def compactify_keys(self):
        """fill the holes in index keys"""
        store = dict()
        for i, (_, elem) in enumerate(sorted(self.items())):
            store[i] = elem
            elem.index = i
        self.clear()
        self.update(store)


class Topology:
    def __init__(self):
        self.halfedges = ElemCollection(Halfedge)
        self.vertices = ElemCollection(Vertex)
        self.edges = ElemCollection(Edge)
        self.faces = ElemCollection(Face)

    def build(self, n_vertices: int, indices: np.ndarray):
        # TODO: P1 -- complete this function
        """
        This will be the primary function for generating your halfedge data structure. As
        the name suggests, the central element of this data structure will be the Halfedge.
        Halfedges are related to each other through two operations `twin` and `next`.
        Halfedge.twin returns the halfedge the shares the same edge as the current Halfedge,
        but is oppositely oriented (e.g. if halfedge H points from vertex A to vertex B,
        then H.twin points from vertex B to vertex A). Halfedge.next returns the next
        halfedge within the same triangle in the same orientation (e.g. given triangle ABC,
        if halfedge H goes A->B, then H.next goes B -> C).

        With these properties alone, every halfedge can be associated with a specific face,
        vertex, and edge. Thus, in your implementation every halfedge H should be assigned a
        Face, Vertex, and Edge element as attributes. Likewise, every Face, Vertex, and Edge
        element should be assigned a halfedge H. Note that this relationship is not 1:1, so
        that there are multiple valid halfedges you can assign to each Face, Vertex, and
        Edge. The choice is not important. As long as the orientation of the elements are
        consistent across the mesh, then your implementation should work.

        Arguments:
        - n_vertices: how many vertices in the vertices array
        - indices: int array of shape (n_faces, 3); each row [i, j, k] is a triangular face
        made of vertices with indices i, j, k. The vertices' positions in space are not
        important for building a halfedge structure, only their connectivity.

        ======== VERY IMPORTANT =======
        In order for your implementation to pass our checks, you MUST allocate
        faces/halfedges in the following order, for each row (face array) in `indices` array:
            - If a face array contains vertex indices [i,j,k], then allocate halfedges/edges
              in the order (i,j), (j,k), (k, i)
            - If an edge has already been encountered, then set the new halfedge as the
              `twin` of the existing halfedge

        You should use self.halfedges.allocate() when creating a halfedge (and same for
        self.faces, self.vertices, self.edges);  it will create, keep track of, and return a
        new instance of the corresponding primitive with an assigned, incrementing index but
        nothing else. This index is also its key in the corresponding ElemCollection
        (self.halfedges, self.faces, self.vertices, self.edges). You'll set its other
        properties (the face, vertex, edge, next, twin if a Halfedge, and a halfedge if a
        Face, Vertex, or Edge).
        """
        # Pre-allocating all our vertices
        for i in range(n_vertices):
            self.vertices.allocate()
    
        # Dictionary of the structure {(v_index_1, v_index_2): edge}.
        visited_edges = {}

        for i, face in enumerate(indices):
            f = self.faces.allocate()
            # List of halfedges for each face
            face_hes = []
            # Allocating 1 halfedge for each edge in our face
            for j in range(3):
                face_hes.append(self.halfedges.allocate())
            # Iterating over each vertex in a face
            # For vertices [i, j, k], we allocate the edges in the
            # following order: (i,j), (j,k), (k,i).
            for k, vertex_index in enumerate(face):
                he = face_hes[k]
                # Assigning the next halfedge
                he.next = face_hes[(k+1)%3]                
                # Set fields for halfedge
                v = self.vertices[vertex_index]
                # Assigning the vertex (which here is just one number) to our halfedge
                he.vertex = v  
                he.face = f
                # Linking vertex to halfedge
                v.halfedge = he
                f.halfedge = he

                '''
                Keeping track of which edges we've already created a halfedge for.
                We are sorting the the list so that (a, b) and (b, a) will both index
                to the same edge.
                We convert this list to a tuple so that we can use it as an index
                for our visted_edges hash_map (dictionary).
                '''
                edge_vertices = tuple(sorted([vertex_index, face[(k+1)%3]]))

                if edge_vertices in visited_edges:
                    e = visited_edges[edge_vertices]
                    # Setting he and e.halfedge as twins of each other
                    he.twin = e.halfedge
                    e.halfedge.twin = he
                    he.edge = e
                else:
                    e = self.edges.allocate()
                    e.halfedge = he
                    he.edge = e
                    visited_edges[edge_vertices] = e

        self.thorough_check()

    def compactify_keys(self):
        self.halfedges.compactify_keys()
        self.vertices.compactify_keys()
        self.edges.compactify_keys()
        self.faces.compactify_keys()

    def export_halfedge_serialization(self):
        """
        this provides the unique, unambiguous serialization of the halfedge topology
        i.e. one can reconstruct the mesh connectivity from this information alone
        It can be used to track history, etc.
        """
        data = []
        for _, he in sorted(self.halfedges.items()):
            data.append(he.serialize())
        data = np.array(data, dtype=np.int32)
        return data

    def export_face_connectivity(self):
        face_indices = []
        for inx, face in self.faces.items():
            vs_of_this_face = []
            if face.halfedge is None:
                continue
            for vtx in face.adjacentVertices():
                vs_of_this_face.append(vtx.index)
            assert len(vs_of_this_face) == 3
            face_indices.append(vs_of_this_face)
        return face_indices

    def export_edge_connectivity(self):
        conn = []
        for _, edge in self.edges.items():
            if edge.halfedge is None:
                continue
            v1 = edge.halfedge.vertex
            v2 = edge.halfedge.twin.vertex
            conn.append([v1.index, v2.index])
        return conn

    def hasNonManifoldVertices(self):
        all_adjacents = {v.index: set() for v in self.vertices.values()}

        # Grabbing the adjacent vertices of vertex
        # by adding pairs of adjacent vertices to our adjacency set.
        for he in self.halfedges.values():
            u = he.vertex
            v = he.tip_vertex()
            all_adjacents[u.index].add(v)
            all_adjacents[v.index].add(u)

        # Checking if any vertex has a different set of adjacent
        # neighbors than those could be accessed on a mainfold surface.
        for v in self.vertices.values():
            # Type-casting adjacentVertices sicne it returns a list. 
            if all_adjacents[v.index] != set(v.adjacentVertices()):
                return True
    
        return False

    def hasNonManifoldEdges(self):
        halfedges_per_edge = {e.index : 0 for e in self.edges.values()}

        for he in self.halfedges.values():
            e = he.edge
            halfedges_per_edge[e.index] += 1

            # This edge is associated with more than two halfedges,
            # and thus also more than two faces, making it a non-manifold edge.
            if halfedges_per_edge[e.index] > 2:
                return True
        
        return False

    def thorough_check(self):
        if (
            len(self.halfedges) == 0
            or len(self.vertices) == 0
            or len(self.faces) == 0
            or len(self.edges) == 0
        ):
            print(
                f"[thorough_check] Topology is incomplete. You need to allocate halfedge, vertex, face, and edge elements in order for the checker to work."
            )
            return

        def check_indexing(src_dict):
            for inx, v in src_dict.items():
                assert inx == v.index

        check_indexing(self.halfedges)
        check_indexing(self.vertices)
        check_indexing(self.edges)
        check_indexing(self.faces)

        # Check full halfedge coverage across all mesh elements
        self._check_edges()

        try:
            self._check_verts()
        except NotImplementedError as e:
            print(
                f"[thorough_check] _check_verts crashed likely because Vertex.adjacentHalfedges has not been implemented. The error was\n{e}"
            )

        try:
            self._check_faces()
        except NotImplementedError as e:
            print(
                f"[thorough_check] _check_faces crashed likely because Face.adjacentHalfedges has not been implemented. The error was\n{e}"
            )

    def _check_verts(self):
        encountered_halfedges = []
        for inx, v in self.vertices.items():
            hes = []
            for he in v.adjacentHalfedges():
                assert he.vertex == v
                hes.append(he)
            encountered_halfedges.extend([elem.index for elem in hes])
        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), (
            "must cover all halfedges"
        )

    def _check_edges(self):
        encountered_halfedges = []
        for inx, e in self.edges.items():
            he = e.halfedge
            twin = he.twin

            hes = [he, twin]
            n = len(hes)

            for i, he in enumerate(hes):
                assert he.edge == e
                assert he.twin == hes[(i + 1) % n]

            encountered_halfedges.extend([elem.index for elem in hes])

        encountered_halfedges = set(encountered_halfedges)
        assert encountered_halfedges == set(self.halfedges.keys()), (
            "must cover all halfedges"
        )

    def _check_faces(self):
        encountered_halfedges = []
        for inx, f in self.faces.items():
            hes = []
            for he in f.adjacentHalfedges():
                hes.append(he)

            n = len(hes)
            for i, he in enumerate(hes):
                assert he.face == f
                assert he.next == hes[(i + 1) % n]

            encountered_halfedges.extend([elem.index for elem in hes])

        encountered_halfedges = set(encountered_halfedges)
        target_halfedges = {k for k, v in self.halfedges.items()}
        assert encountered_halfedges == target_halfedges, f"must cover all halfedges"
