import numpy as np
from meshing.io import PolygonSoup
from meshing.mesh import Mesh

if __name__ == "__main__":

    def load_bunny_and_check():
        # Initializes a PolygonSoup object from an obj file, which reads in the obj
        # and fills out attributes vertices and indices (triangle faces)
        soup = PolygonSoup.from_obj("bunny.obj")
        # Initialize your mesh object
        mesh = Mesh(soup.vertices, soup.indices)
        mesh.view_basic()

        # This is the checker function that is called at the end of topology.build()
        # Currently, this will throw.
        # It will work partly once you actually start implementing topology.build()
        # (P1, topology.py) and fully work once you finish that and the
        # Vertex.adjacentHalfedges, Face.adjacentHalfedges functions (P2, primitive.py)
        # Even before you get to P2, you should make sure thorough_check runs up to
        # and including _check_edges successfully.
        mesh.topology.thorough_check()
        return mesh

    # Run these example functions once you've finished their prerequisites as commented
    # Note that Mesh.get_3d_pos is P4 but doesn't depend on other P4 functions;
    # you can write this (very simple) function to get the viewer working early.

    def example_halfedge0():
        """
        This function won't work until you implement the functions
        - Topology.build in topology.py (P1)
        - Mesh.get_3d_pos for the viewer (P4, but doesn't depend on other P4
        functions; you can write this (very simple) function to get the viewer
        working early
        - Vertex.adjacentHalfedges, Face.adjacentHalfedges, Face.adjacentVertices,
        Halfedge.prev, Halfedge.tip_vertex (tip_vertex is for the viewer) in
        primitive.py (P2)
        """
        mesh = load_bunny_and_check()
        # Examples of some of the accessor functions you need to implement for P2,
        # along with how you can visualize/check them
        # Get first halfedge in mesh
        he = mesh.topology.halfedges[0]
        # Previous of next halfedge is just original halfedge
        assert he == he.next.prev()
        # Highlight the faces associated with the current halfege and its twin
        twin_face = he.twin.face
        h_face = he.face
        mesh.view_with_topology(
            highlight_faces=(twin_face, h_face),
            highlight_halfedges=(he,),
        )

    def example_onering():
        """
        This function won't work until you implement the functions
        - Topology.build in topology.py (P1)
        - Mesh.get_3d_pos for the viewer (P4, but doesn't depend on other P4
        functions; you can write this (very simple) function to get the viewer
        working early
        - Vertex.adjacentHalfedges, Face.adjacentHalfedges, Face.adjacentVertices in primitive.py (P2)
        - Vertex.adjacentEdges in primitive.py (P2)
        - Edge.two_vertices in primitive.py, for the viewer (P2)
        """
        mesh = load_bunny_and_check()
        he = mesh.topology.halfedges[0]
        # Plot all the edges extending out from the current halfedge vertex
        vertex_onering_edges = list(he.vertex.adjacentEdges())
        mesh.view_with_topology(highlight_edges=vertex_onering_edges)

    def example_export():
        """
        This function won't work until you implement the functions
        - Topology.build in topology.py (P1)
        - Vertex.adjacentHalfedges, Face.adjacentHalfedges, Face.adjacentVertices in primitive.py (P2)
        - Face.adjacentVertices in primitive.py (P2)
        """
        mesh = load_bunny_and_check()
        # Exports current mesh vertices, faces, and edges
        vertices, faces, edges = mesh.export_soup()

        # Save current mesh to an obj
        mesh.export_obj("your_shape.obj")

    def example_smoothing():
        """
        This is for testing P5 laplacian smoothing on the bunny mesh.
        You should have finished P1, P2, P3, P4 and begun working on P5 for this.
        """
        mesh = load_bunny_and_check()
        mesh.smoothMesh(n=5)
        mesh.view_basic()
        mesh.export_obj("p5.obj")

    def example_collapse_simple():
        """
        This is for testing P6, a single edge collapse on a simple test mesh.
        You should have finished P1, P2, P3, P4 and begun working on P6 for this.
        """
        simple_mesh_soup = PolygonSoup.from_obj("single_edge_collapse.obj")
        simple_mesh = Mesh(simple_mesh_soup.vertices, simple_mesh_soup.indices)
        simple_mesh.view_with_topology(highlight_edges=[simple_mesh.topology.edges[0]])
        simple_mesh.collapse([0])  # collapse edge with index 0
        simple_mesh.view_with_topology()

    def example_collapse_simple_cube():
        """
        This is for testing P6, a single degree-3 edge collapse on a simple cube mesh.
        You should have finished P1, P2, P3, P4 and begun working on P6 for this.
        """
        simple_mesh_soup = PolygonSoup.from_obj("cube.obj")
        simple_mesh = Mesh(simple_mesh_soup.vertices, simple_mesh_soup.indices)
        simple_mesh.view_with_topology(highlight_edges=[simple_mesh.topology.edges[0]])
        simple_mesh.collapse([0])  # collapse edge with index 0
        simple_mesh.view_with_topology()

    def example_collapses():
        """
        This is for testing P6 edge collapses on the bunny mesh, with the given
        sequence of edges.
        You should have finished P1, P2, P3, P4 and begun working on P6 for this.
        """
        edge_ids = np.load("bunny_collapses.npy")
        mesh = load_bunny_and_check()
        mesh.collapse(edge_ids)
        mesh.view_with_topology()
        mesh.export_obj("p6.obj")

    def example_collapses_with_link():
        """
        This is for testing Extra Credit edge collapses with link condition on
        the bunny mesh, with the given sequence of edges many of which violate
        the link condition.
        You should have finished P1, P2, P3, P4, P6, and begun working on the
        extra credit for this.
        """
        edge_ids = np.load("bunny_collapses_link.npy")
        mesh = load_bunny_and_check()
        mesh.collapse_with_link_condition(edge_ids)
        mesh.view_with_topology()
        mesh.export_obj("ec.obj")

    ## run one of these functions at a time per script run
    load_bunny_and_check()
    # example_halfedge0()
    # example_onering()
    # example_export()
    # example_smoothing()
    # example_collapse_simple()
    # example_collapse_simple_cube()
    # example_collapses()
    # example_collapses_with_link()
