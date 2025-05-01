from typing import Sequence, Tuple, Union
import numpy as np
from . import Halfedge, Edge, Vertex, Face, Topology

"""
NOTE: We will NOT deal with boundary loops

TODO P4 -- complete the functions
- Mesh.get_3d_pos, Mesh.vector, Mesh.faceNormal

For P5 and P6, make changes to edit.py, whose functions are called from here.
"""


class Mesh:
    def __init__(self, vertices: np.ndarray, face_indices: np.ndarray):
        self.vertices = vertices
        self.indices = face_indices
        self.topology = Topology()
        self.topology.build(len(vertices), face_indices)

    def export_soup_with_index_key_maps(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        NOTE you should not need to use this function in your implementation
        (just use export_soup if you need packed vertices and faces), but this
        may come in handy if you wish to dive deep into debugging

        if you're curious:
            the topology's ElemCollections are almost like 'sparse arrays';
            there may be holes in the keys (i.e. keys which were once present
            but got deleted, leaving a gap in the range of valid "array
            indices"), which would have previously held an element that later
            got deleted (e.g. due to an edit like edge collapse). All keys
            past and present are valid indices into self.vertices, which may
            have unreferenced vertices corresponding to the holes/deleted keys
            in the ElemCollection.
            A Primitive object's `index` property == its key in the ElemCollection.

            Exporting the soup will pack indices and trim the vertices array
            (which those packed indices point into) to eliminate these holes for
            operations that expect regular, packed vertices and indices arrays.

        returns
        - vertices  (n_packed_vertices, 3) float array of vertex coordinates
        - face__v_packed_idx (n_packed_faces, 3) int array of indices into this
            packed vertices array, each row a face
        - edges__v_packed_idx (n_packed_edges, 2) int array of indices into this
            packed vertices array, each row an edge
        - packed2key (n_packed_vertices,)  int array where entry i is the original
            ElemCollection key for the ith vertex in the packed vertices array
        - key2packed (len(self.vertices),) int array where entry i is the
            index in the packed vertices array for the vertex originally at key i
            in the ElemCollection
        """
        # face and edge arrays that use the vertex's ElemCollection keys (the
        # vertex.index field) rather than packed array indices
        faces__v_keys = np.array(self.topology.export_face_connectivity(), dtype=np.uint32)
        edges__v_keys = np.array(self.topology.export_edge_connectivity(), dtype=np.uint32)

        packed2key = np.array(sorted(self.topology.vertices.keys()), dtype=np.uint32)
        n_packed_vertices = len(packed2key)
        vertex_packed_inds = np.arange(n_packed_vertices, dtype=np.uint32)
        vertices = self.vertices[packed2key]

        key2packed = np.zeros(len(self.vertices), dtype=np.uint32)
        key2packed[packed2key] = vertex_packed_inds

        faces__v_packed_idx = key2packed[faces__v_keys]
        edges__v_packed_idx = key2packed[edges__v_keys]
        return (vertices, faces__v_packed_idx, edges__v_packed_idx, packed2key, key2packed)

    def export_soup(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns (vertices, face_indices, edge_indices)
        - vertices  (n_packed_vertices, 3) float array of vertex coordinates
        - face_indices (n_packed_faces, 3) int array of indices into vertices,
            each row a face
        - edge_indices (n_packed_edges, 2) int array of indices into vertices,
            each row an edge
        """
        vertices, face_indices, edge_indices, _, _ = self.export_soup_with_index_key_maps()
        return vertices, face_indices, edge_indices

    def get_3d_pos(self, v: Vertex) -> np.ndarray:
        """Given a vertex primitive, return the position coordinates"""
        vindex = v.index
        return self.vertices[vindex]

    def vector(self, h: Halfedge) -> np.ndarray:
        """Given a halfedge primitive, return the vector"""
        return np.array([self.vertices[h.vertex], self.vertices[h.tip_vertex()]])

    # TODO: P4 -- complete this
    def faceNormal(self, f: Face) -> np.ndarray:
        """Given a face primitive, compute the unit normal"""
        # We only need three edges to find the unit normal vector
        he1 = f.halfedge
        he2 = f.halfedge.next
        he3 = f.halfedge.next.next

        # 3 3D points on the face
        p1 = self.get_3d_pos(he1.vertex)
        p2 = self.get_3d_pos(he2.vertex)
        p3 = self.get_3d_pos(he3.vertex)

        # 2 vectors on the face
        v1 = p1 - p2
        v2 = p3 - p2

        n = np.cross(v1, v2)
        # Normalizing the normal vector
        n /= np.linalg.norm(n)

        return n

    # TODO: P5 (make changes in edit.py)
    def smoothMesh(self, n=5):
        """Laplacian smooth mesh n times"""
        from . import LaplacianSmoothing

        LaplacianSmoothing(self, n).apply()

    # TODO: P6 (make changes in edit.py)
    def collapse(self, edge_ids: Sequence[int]):
        """Edge collapse without link condition check"""
        from . import EdgeCollapse

        for e_id in edge_ids:
            edt = EdgeCollapse(self, e_id)
            edt.apply()

    # TODO: Extra credit (make changes in edit.py)
    def collapse_with_link_condition(self, edge_ids: Sequence[int]):
        """Extra credit: collapse with link condition check"""
        from . import EdgeCollapseWithLink

        for e_id in edge_ids:
            edt = EdgeCollapseWithLink(self, e_id)
            edt.apply()

    def view_basic(self):
        """
        Simple mesh viewer using polyscope.
        Directly visualizes self.vertices, self.indices WITHOUT any consideration
        of mesh.topology. This is valid as long as you don't perform any edits
        to the topology structure.

        Vertex location updates that make changes only to self.vertices are
        reflected correctly, such as LaplacianSmoothing, but not EdgeCollapse
        which modifies the topology and makes this visualization not correct.
        For such situations, try view_with_topology, which also has abilities to
        highlight primitive objects in the visualization.
        """
        import polyscope as ps

        ps.init()
        ps.register_surface_mesh("mesh", self.vertices, self.indices)
        ps.show()

    def view_with_topology(
        self,
        highlight_vertices: Sequence[Vertex] = (),
        highlight_edges: Sequence[Edge] = (),
        highlight_halfedges: Sequence[Halfedge] = (),
        highlight_faces: Sequence[Face] = (),
        clear_existing_structures: bool = False,
    ):
        """
        Mesh viewer using polyscope that reflects the mesh implied by self.topology.

        By default we'll always clear existing registered polyscope structures first.
        If clear_existing_structures (default True) is False, then we won't clear them.

        highlight_* arguments will register extra Polyscope structures
        corresponding to the primitive objects you give, annotated with their
        original ElemCollection key (their primitive.index) value.
        Note that
        - highlight_vertices won't work until you implement Mesh.get_3d_pos (P4)
        - highlight_edges and highlight_halfedges won't work until you implement
          two_vertices for Edge (P2) and Mesh.get_3d_pos (P4)
        - highlight_halfedges won't work until you implement
          tip_vertex for Halfedge (P2) and Mesh.get_3d_pos (P4)
        - highlight_* won't work at all until you finish Topology.build (P1)

        Note that Mesh.get_3d_pos is P4 but doesn't depend on other P4
        functions; you can write this (very simple) function to get the viewer
        working early.

        Read the docstring of export_soup_with_index_key_maps() for more info.

        Primitives that are highlighted get assigned a scalar quantity in the Polyscope
        visualization called "key in topology", which corresponds to their
        primitive.index value, which is also their key in the corresponding
        ElemCollection in mesh.topology. These scalars are helpfully colormapped
        by Polyscope for nice viewing; you can click on the objects in Polyscope to see
        their quantities (i.e. their keys visualized here) on the top right window.

        (There may also be a printout about elements that "have an index" in
        their ElemCollection but no longer exist in the mesh.topology halfedge
        structure; this happens if they are deleted from the ElemCollection,
        since ElemCollection never reuses a deleted object's key for new objects
        unless compactified.)

        For faces specifically: since highlighting faces doesn't involve any new
        Polyscope structures and just colors the mesh's faces, non-highlighted
        faces are assigned the scalar -1, and highlighted faces get assigned
        their key as their scalar in the viz in Polyscope.
        """
        import polyscope as ps

        ps.init()
        if clear_existing_structures:
            ps.remove_all_structures()

        vs, fs, es, packed2key, key2packed = self.export_soup_with_index_key_maps()
        ps_mesh = ps.register_surface_mesh("mesh", vs, fs, edge_width=1)
        ps_mesh.add_scalar_quantity(
            "vertex key in topology", packed2key, datatype="categorical"
        )

        if highlight_vertices:
            highlight_vertices_existornot = tuple(
                (v, v.index in self.topology.vertices) for v in highlight_vertices
            )
            pcloud = ps.register_point_cloud(
                "vertices of interest",
                np.stack(
                    [self.get_3d_pos(v) for (v, exists) in highlight_vertices_existornot],
                    axis=0,
                ),
            )
            # visualize the keys which may not be the packed indices from exported soup
            pcloud.add_scalar_quantity(
                "key in topology",
                np.array([v.index for v in highlight_vertices]),
                enable=True,
                datatype="categorical",
            )
            # show and report the ones that don't exist
            pcloud.add_scalar_quantity(
                "exists in topology",
                np.array(
                    [exists for (v, exists) in highlight_vertices_existornot],
                    dtype=np.uint32,
                ),
                datatype="categorical",
            )
            highlight_vertices_noexist = tuple(
                v.index for (v, exists) in highlight_vertices_existornot if not exists
            )
            if highlight_vertices_noexist:
                print(
                    f"These requested highlight vertices with the following vertex.index keys don't exist in mesh.topology: {highlight_vertices_noexist}"
                )

        def _viz_edgy(prims: Sequence[Union[Halfedge, Edge]], as_edge: bool):
            edgy = "edge" if as_edge else "halfedge"
            _keys = []
            _keys_noexist = []
            _nodes = []
            _vecs = []  # for halfedge vector visualization
            for prim in prims:
                key = prim.index
                _keys.append(key)
                exists_in_topo = key in (
                    self.topology.edges
                    if isinstance(prim, Edge)
                    else self.topology.halfedges
                )
                if not exists_in_topo:
                    _keys_noexist.append(key)
                two_vertices = (
                    np.array([self.get_3d_pos(v) for v in prim.two_vertices()])
                    if isinstance(prim, Edge)
                    else np.array(
                        [self.get_3d_pos(prim.vertex), self.get_3d_pos(prim.tip_vertex())]
                    )
                )
                _nodes.append(two_vertices)
                if not as_edge:
                    _vecs.append(
                        np.array(
                            [
                                two_vertices[1] - two_vertices[0],
                                np.zeros_like(two_vertices[0]),
                            ]
                        )
                    )

            # need polyscope 2.4.0 for the 'segments' convenience
            # (if a halfedge and its twin are both present, there will most likely be
            # zfighting! we are not handling this explicitly yet for now)
            cunet = ps.register_curve_network(
                f"{edgy}s of interest",
                np.concatenate(_nodes, axis=0),
                edges="segments",
                enabled=True,
            )
            cunet.add_scalar_quantity(
                "key in topology",
                np.array(_keys, dtype=np.uint32),
                defined_on="edges",
                datatype="categorical",
            )
            if not as_edge:
                cunet.add_vector_quantity(
                    "halfedge vector",
                    np.concatenate(_vecs, axis=0),
                    defined_on="nodes",
                    vectortype="ambient",
                    enabled=True,
                )

            if _keys_noexist:
                print(
                    f"These requested highlight {edgy}s with the following {edgy}.index key don't exist in mesh.topology: {_keys_noexist}"
                )

        if highlight_edges:
            _viz_edgy(highlight_edges, as_edge=True)
        if highlight_halfedges:
            _viz_edgy(highlight_halfedges, as_edge=False)
        if highlight_faces:
            # just add a scalar to the ps_mesh; unselected faces get -1
            face_keys = np.full((len(fs),), -1)
            face_keys_highlight__ = [f.index for f in highlight_faces]
            face_keys_highlight = np.array(face_keys_highlight__, dtype=np.uint32)
            face_keys[face_keys_highlight] = face_keys_highlight
            face_keys_noexist = tuple(
                key for key in face_keys_highlight__ if key not in self.topology.faces
            )
            ps_mesh.add_scalar_quantity(
                "faces of interest", face_keys, defined_on="faces", datatype="categorical"
            )
            if face_keys_noexist:
                print(
                    f"These requested highlight faces with the following face.index keys don't exist in mesh.topology: {face_keys_noexist}"
                )

        ps.show()

    def export_obj(self, path):
        vertices, faces, edges = self.export_soup()
        with open(path, "w") as f:
            for vi, v in enumerate(vertices):
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for face_id in range(len(faces) - 1):
                f.write(
                    "f %d %d %d\n"
                    % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1)
                )
            f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
            for edge in edges:
                f.write("\ne %d %d" % (edge[0] + 1, edge[1] + 1))
