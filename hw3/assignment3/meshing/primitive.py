from typing import Optional, Iterable, Tuple
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
        # TODO: P2 -- complete this function
        """Return previous halfedge"""
        raise NotImplementedError("TODO (P2)")

    def tip_vertex(self) -> "Vertex":
        # TODO: P2 -- complete this function
        """Return vertex on the tip of the halfedge"""
        raise NotImplementedError("TODO (P2)")

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
        # TODO: P2 -- complete this function
        """
        return the two incident vertices of the edge
        note that the incident vertices are ambiguous to ordering
        """
        raise NotImplementedError("TODO (P2)")


class Face(Primitive):
    """
    Has halfedge and index (see Primitive base class). These fields are filled
    after __init__ (the ElemCollection.allocate function in topology.py will
    assign an index, and you will handle the rest)
    """

    def adjacentHalfedges(self) -> Iterable[Halfedge]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent halfedges"""
        raise NotImplementedError("TODO (P2)")

    def adjacentVertices(self) -> Iterable["Vertex"]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent vertices"""
        raise NotImplementedError("TODO (P2)")

    def adjacentEdges(self) -> Iterable[Edge]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent edges"""
        raise NotImplementedError("TODO (P2)")

    def adjacentFaces(self) -> Iterable["Face"]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent faces"""
        raise NotImplementedError("TODO (P2)")


class Vertex(Primitive):
    """
    Has halfedge and index (see Primitive base class). These fields are filled
    after __init__ (the ElemCollection.allocate function in topology.py will
    assign an index, and you will handle the rest)
    """

    def degree(self) -> int:
        # TODO: P2 -- complete this function
        """Return vertex degree: # of incident edges"""
        raise NotImplementedError("TODO (P2)")

    def isIsolated(self) -> bool:
        # because self.halfedge will throw if __halfedge is None, but we don't
        # want this to throw, we'll just use __halfedge directly for this one
        return self.__halfedge is None

    def adjacentHalfedges(self) -> Iterable[Halfedge]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent halfedges"""
        raise NotImplementedError("TODO (P2)")

    def adjacentVertices(self) -> Iterable["Vertex"]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent vertices"""
        raise NotImplementedError("TODO (P2)")

    def adjacentEdges(self) -> Iterable[Edge]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent edges"""
        raise NotImplementedError("TODO (P2)")

    def adjacentFaces(self) -> Iterable[Face]:
        # TODO: P2 -- complete this function
        """Return an iterable of adjacent faces"""
        raise NotImplementedError("TODO (P2)")
