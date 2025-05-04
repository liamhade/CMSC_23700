My approach to anyone starting out on this project would be to make sure they
draw everything out -- this project is inherently visual, and thus it is
most helpful that, before you design the code, you have an intuitive understanding
of the visuals.

Besides edge-collapse, checking for non-manifold behavior is likely the trickiest part of this project.
In the approach of this part of the project, I found it most helpful to use the functions
that I already coded early, such as for adjacentEdges() and adjacentFaces() to figure
out what the expected behavior of a manifold surface would be, and then compare
that to the actual adjacency behavior of the surface.

For edge-collapse, I would say the trickiest part is handling the halfedges that are leftover
after their twins and edges are deleted. Additionally, you're going to have to delete an edge in each
face, but which edge you choose is arbitrary -- just make sure that the edge you choose is consistent
and that you understand how to navigate from that edge.