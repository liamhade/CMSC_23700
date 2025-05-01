from typing import Optional, Iterable, Tuple, List
import numpy as np

"""
TODO P2 -- complete the functions
- Halfedge.prev, Halfedge.tip_vertex
- Edge.two_vertices
- Face.adjacentHalfedges, Face.adjacentVertices, Face.adjacentEdges, Face.adjacentFaces
- Vertex.degree, Vertex.adjacentHalfedges, Vertex.adjacentVertices, Vertex.adjacentEdges, Vertex.adjacentFaces
"""


class UninitializedHalfedgePropertyError(BaseException):
    """
    an exception thrown when trying to get a value from an uninitialized halfedge property
    """

    pass


class Primitive:
    def __init__(self):
        # NOTE ignore these private fields, you should use the field names without the __ in
        # front. These getters do None-checking to make sure None isn't silently returned.
        # You should get and set values using prim.halfedge, prim.index
        self.__halfedge: Optional["Halfedge"] = None
        self.__index: Optional[int] = None

    @property
    def halfedge(self) -> "Halfedge":
        if self.__halfedge is None:
            raise UninitializedHalfedgePropertyError(
                f"malformed halfedge structure: {self.__class__.__qualname__}.halfedge gave None"
            )
        return self.__halfedge

    @halfedge.setter
    def halfedge(self, value: "Halfedge"):
        self.__halfedge = value

    @property
    def index(self) -> int:
        """A primitive's index is its key in the corresponding ElemCollection in the Topology"""
        if self.__index is None:
            raise UninitializedHalfedgePropertyError(
                f"malformed halfedge structure: {self.__class__.__qualname__}.index gave None"
            )
        return self.__index

    @index.setter
    def index(self, value: int):
        self.__index = value

    def __str__(self) -> str:
        return str(self.index)

    def __repr__(self) -> str:
        return str(self)


class Halfedge(Primitive):
    def __init__(self):
        # NOTE ignore these private fields, you should use the field names
        # without the __ in front. Get values from and assign values to
        # halfedge.vertex, halfedge.edge, halfedge.twin, and so on. The idea is
        # that these private fields start uninitialized (as None) but getting
        # the non-private fields should never return None (and will throw if the
        # field is None, rather than silently returning None, which is bad)
        self.__vertex: Optional["Vertex"] = None
        self.__edge: Optional["Edge"] = None
        self.__face: Optional["Face"] = None
        self.__next: Optional["Halfedge"] = None
        self.__twin: Optional["Halfedge"] = None
        self.onBoundary: bool = False
        self.__index: Optional[int] = None
        # an ID between 0 and |H| - 1, where |H| is the number of halfedges in a mesh.

    ############## boilerplate
    # these property getters and setters are just for None-checking upon
    # accessing the index, vertex, edge, face, next, twin attributes to make
    # sure they were all initialized and a None isn't silently returned.
    # You should use these attributes without the __ (those __attribs are private)
    @property
    def index(self) -> int:
        if self.__index is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.index gave None"
            )
        return self.__index

    @index.setter
    def index(self, value: int):
        self.__index = value

    @property
    def vertex(self) -> "Vertex":
        if self.__vertex is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.vertex gave None"
            )
        return self.__vertex

    @vertex.setter
    def vertex(self, value: "Vertex"):
        self.__vertex = value

    @property
    def edge(self) -> "Edge":
        if self.__edge is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.edge gave None"
            )
        return self.__edge

    @edge.setter
    def edge(self, value: "Edge"):
        self.__edge = value

    @property
    def face(self) -> "Face":
        if self.__face is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.face gave None"
            )
        return self.__face

    @face.setter
    def face(self, value: "Face"):
        self.__face = value

    @property
    def next(self) -> "Halfedge":
        if self.__next is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.next gave None"
            )
        return self.__next

    @next.setter
    def next(self, value: "Halfedge"):
        self.__next = value

    @property
    def twin(self) -> "Halfedge":
        if self.__twin is None:
            raise UninitializedHalfedgePropertyError(
                "malformed halfedge structure: Halfedge.twin gave None"
            )
        return self.__twin

    @twin.setter
    def twin(self, value: "Halfedge"):
        self.__twin = value

    ############## end boilerplate

    def prev(self) -> "Halfedge":
        """Return previous halfedge"""
        he = self
        # Incrementing until the next halfedge is our starting halfedge.
        while he.next != self:
            he = he.next
        return he

    def tip_vertex(self) -> "Vertex":
        """Return vertex on the tip of the halfedge"""
        return self.next.vertex

    def serialize(self):
        return (
            self.index,
            self.vertex.index,
            self.edge.index,
            self.face.index,
            self.next.index,
            self.twin.index,
        )


class Edge(Primitive):
    """
    Has halfedge and index (see Primitive base class). These fields are filled
    after __init__ (the ElemCollection.allocate function in topology.py will
    assign an index, and you will handle the rest)
    """

    def two_vertices(self) -> Tuple["Vertex", "Vertex"]:
        """
        return the two incident vertices of the edge
        note that the incident vertices are ambiguous to ordering
        """
        return (self.halfedge.vertex, self.halfedge.tip_vertex())


class Face(Primitive):
    """
    Has halfedge and index (see Primitive base class). These fields are filled
    after __init__ (the ElemCollection.allocate function in topology.py will
    assign an index, and you will handle the rest)
    """

    def adjacentHalfedges(self) -> Iterable[Halfedge]:
        """Return an iterable of adjacent halfedges"""
        halfedges = []
        he = self.halfedge
        while he not in halfedges:
            halfedges.append(he)
            he = he.next
        return halfedges

    def adjacentVertices(self) -> Iterable["Vertex"]:
        """Return an iterable of adjacent vertices"""

        '''
        We want to check the adjacent vertices that are clockwise
        and anti-clockwise, since we don't yet know if the surfaces
        in this program are manifold.

        If we could be certain that they were manifold, then checking
        either direction would suffice.

        Might be hepful to draw out the faces with halfedges if this
        explanation seems confusing.
        '''
        vertices = []
        he = self.halfedge
        v  = he.vertex
        while v not in vertices:
            vertices.append(v)
            # Stepping to the next halfedge
            he = he.next 
            v  = he.vertex
        return vertices

    def adjacentEdges(self) -> Iterable[Edge]:
        """Return an iterable of adjacent edges"""
        edges = []
        he = self.halfedge
        e  = he.edge
        while e not in edges:
            edges.append(e)
            he = he.next
            e = he.edge
        return edges

    def adjacentFaces(self) -> Iterable["Face"]:
        """Return an iterable of adjacent faces"""
        faces = []
        he = self.halfedge
        # We have to do twin, since otherwise we'd be fetching
        # the face that we're on right now.
        f  = he.twin.face
        while f not in faces:
            faces.append(f)
            he = he.next
            f = he.twin.face
        return faces

class Vertex(Primitive):
    """
    Has halfedge and index (see Primitive base class). These fields are filled
    after __init__ (the ElemCollection.allocate function in topology.py will
    assign an index, and you will handle the rest)
    """

    def degree(self) -> int:
        """Return vertex degree: # of incident edges"""
        return len(self.adjacentVertices())


    def isIsolated(self) -> bool:
        # because self.halfedge will throw if __halfedge is None, but we don't
        # want this to throw, we'll just use __halfedge directly for this one
        return self.__halfedge is None

    def adjacentHalfedges(self) -> Iterable[Halfedge]:
        """Return an iterable of adjacent halfedges"""
        halfedges = []
        he = self.halfedge
        while he not in halfedges:
            halfedges.append(he)
            he = he.prev().twin
        return halfedges

    def adjacentVertices(self) -> List["Vertex"]:
        """Return an iterable of adjacent vertices"""
        adjacent_vertices = []

        # Adding adjacent vertices in a clockwise manner
        he = self.halfedge
        v  = he.tip_vertex()
        while v not in adjacent_vertices:
            adjacent_vertices.append(v)
            he = he.prev().twin
            v  = he.tip_vertex()
        return adjacent_vertices


    def adjacentEdges(self) -> Iterable[Edge]:
        """Return an iterable of adjacent edges"""
        adjacent_edges = []

        # Adding adjacent edges in a clockwise manner
        he = self.halfedge
        e  = he.edge
        while e not in adjacent_edges:
            adjacent_edges.append(e)
            # Incrementing to the next edge
            he = he.prev().twin
            e  = he.edge
        return adjacent_edges

    def adjacentFaces(self) -> Iterable[Face]:
        """Return an iterable of adjacent faces"""
        adjacent_faces = []

        # Adding adjacent edges in a clockwise manner
        he = self.halfedge
        f  = he.face
        while f not in adjacent_faces:
            adjacent_faces.append(f)
            # Incrementing to the next edge
            he = he.prev().twin
            f  = he.face
        return adjacent_faces

