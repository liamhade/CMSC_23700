from typing import Literal, Sequence, Union
import numpy as np
import arm
import skeleton
from joint import Joint

"""
TODO -- complete the functions
Task 1: rigid_skinning
Task 2: compute_distance_to_line_segment, compute_linear_blended_weights, linear_blend_skinning
"""


# TODO: Task 2 - Subtask 1
#
# Compute the distance of point 'pt' to the line segment
# between point 'vertex0' and 'vertex1'.
#
# pt, vertex0, vertex1 are each a 1D array of shape (3,) representing a 3D point
#
# Hint: You can do this by projecting the point onto the (infinite) line
# and then computing the distance between the projection and the point.
# You need to check whether the projection actually lies within the
# line segment (it might lie outside) - if it doesn't, you need to instead
# return the distance to the closest end point of the segment (i.e. vertex0/vertex1)
def compute_distance_to_line_segment(pt: np.ndarray, vertex0: np.ndarray, vertex1: np.ndarray) -> float:
    line = vertex1 - vertex0
    # Distance along the line that our point falls
    t = ((pt - vertex0) @ line) / np.linalg.norm(line)**2 

    if t > 1:
        # Project is past vertex1
        return np.linalg.norm(pt - vertex1)
    elif t < 0:
        # Projection is past vertex0
        return np.linalg.norm(pt - vertex0)
    else:
        # Projection is on the line segment
        proj = vertex0 + t*line 
        return np.linalg.norm(pt - proj)
    

# SkinMesh represents a triangle mesh that will be skinned with a skeleton
class SkinMesh:
    def __init__(
        self,
        original_positions: np.ndarray,
        indices: np.ndarray,
        transformed_positions: np.ndarray,
        skeleton: skeleton.Skeleton,
        skin_mode: Literal["linear", "rigid"],
    ):
        super().__init__()

        # Original mesh data
        self.original_positions = original_positions  # shape (n_vertices, 3)
        self.indices = indices  # shape (n_faces, 3)

        # Transformed vertex positions
        self.transformed_positions = transformed_positions  # shape (n_vertices, 3)

        # Bind skeleton to skin and compute binding matrices and weights
        self.skeleton = skeleton
        self.skeleton.compute_binding_matrices()
        self.weights = np.zeros((self.get_num_vertices(), self.skeleton.get_num_joints()))
        # self.weights has shape (n_vertices, n_joints), starts all zeros

        self.skin_mode = skin_mode
        if self.skin_mode == "linear":
            self.compute_linear_blended_weights()
        elif self.skin_mode == "rigid":
            self.compute_rigid_weights()
        else:
            raise ValueError(f"Unknown skin_mode {self.skin_mode}")

    # Helper functions

    # Get weight of vertex w.r.t to joint
    def get_vertex_weight(self, vertex_idx: int, joint_idx: int) -> float:
        return self.weights[vertex_idx, joint_idx]

    # Set weight of vertex w.r.t joint
    def set_vertex_weight(self, vertex_idx: int, joint_idx: int, weight: float):
        self.weights[vertex_idx, joint_idx] = weight

    # Returns number of vertices in current mesh
    def get_num_vertices(self) -> int:
        return len(self.original_positions)

    # Get a vertex with index 'idx'
    def get_vertex(self, vertex_idx: int) -> np.ndarray:
        """returns original vertex position (x,y,z) as array of shape (3,)"""
        return self.original_positions[vertex_idx]

    # Set the location of a transformed vertex
    def set_transformed_vertex(
        self, vertex_idx: int, new_pos: Union[np.ndarray, tuple[float, float, float]]
    ):
        self.transformed_positions[vertex_idx] = new_pos

    # Returns the joint associated with vertex in rigid skinning
    def get_rigidly_attached_joint(self, vertex_idx: int) -> Joint:
        weights_for_this_vertex = self.weights[vertex_idx]
        joint_idxs_attached = np.flatnonzero(weights_for_this_vertex)
        assert len(joint_idxs_attached) == 1, (
            "more than one joint attached in rigid skinning is illegal"
        )
        return self.skeleton.get_joint(joint_idxs_attached[0])

    # NOTE: This computes weights for cylinder mesh and assumes ONLY two joints. DON'T use for the arm.
    def compute_rigid_weights(self):
        for i in range(self.get_num_vertices()):
            pos = self.get_vertex(i)
            if pos[0] < 0:
                self.set_vertex_weight(i, 0, 1.0)
                self.set_vertex_weight(i, 1, 0.0)
            else:
                self.set_vertex_weight(i, 0, 0.0)
                self.set_vertex_weight(i, 1, 1.0)

    # TODO: Task 1 - Subtask 2
    # Implement rigid skinning

    # Pseudocode:

    # For each vertex in the mesh  (Hint: use get_num_vertices())
    #   Get rigid joint for vertex (Hint: use get_rigidly_attached_joint)
    #   Compute bone transform     (Hint: The bone transform should transform an (unskinned) vertex position
    #                               1) into the bone's local space computed when it was
    #                                   first bound (i.e. using the binding matrix) and then
    #                               2) back into world space using the current bone transform
    #                                   (i.e. using the world matrix, reflecting its latest angle)
    #   Apply the bone transform you computed to this vertex (Hint: Use get_vertex())
    #   Update the transformed vertex position in the mesh (Hint: Use set_transformed_vertex)
    def rigid_skinning(self):
       for v_idx in range(self.get_num_vertices()):
           joint = self.get_rigidly_attached_joint(v_idx)
           bone_transform = joint.get_world_matrix() @ joint.get_binding_matrix() 
           new_v_position = bone_transform @ self.get_vertex(v_idx)
           self.set_transformed_vertex(v_idx, new_v_position)


    # TODO: Task 2 - Subtask 2
    # Compute smoothly blended vertex weights

    # Pseudocode:
    # For each vertex in the mesh:
    #   For each joint in the skeleton (Hint: use self.skeleton.get_num_joints())
    #       Get world space positions of the joint (Hint: use self.skeleton.get_joint()
    #                                               and joint.get_joint_endpoints())
    #       Compute distance between world space vertex location and joint using compute_distance_to_line_segment
    #       Set the vertex weight to 1/distance^4 (Hint: use set_vertex_weight)
    #
    #   The vertex weights are not yet normalized, so you need to do a second pass
    #   (or incorporate into your first pass) to compute a sum of all of the weights
    #   that you just computed for this vertex. If you choose to do a second
    #   pass over the joints, you can loop over all joints and use get_vertex_weight.
    #
    #   Finally loop over all joints and set the vertex weight for each joint to
    #   the current vertex weight divided by that sum of vertex weights.
    #   Now your vertex's joint weights should sum to one!
    def compute_linear_blended_weights(self):
        # max_w = 0

        # Setting the weights between each vertex and joint
        for v_idx in range(self.get_num_vertices()):
            vertex = self.get_vertex(v_idx)
            for j_idx in range(self.skeleton.get_num_joints()):
                joint = self.skeleton.get_joint(j_idx)
                j_p1, j_p2 = joint.get_endopoints()
                d = compute_distance_to_line_segment(vertex, j_p1, j_p2)
                w = 1/d**4

                self.set_vertex_weight(v_idx, j_idx, w)
            
        # Normalizing the weights so each vertexes weights sum to 1.
        for v_idx in range(self.get_num_vertices()):
            total_v_idx_weight = sum([self.get_vertex_weight(v_idx, j_idx) for j_idx in range(self.skeleton.get_num_joints())])
            for j_idx in range(self.skeleton.get_num_joints()):
                w = self.get_vertex_weight(v_idx, j_idx)
                self.set_vertex_weight(v_idx, j_idx, w / total_v_idx_weight)
            

    # TODO: Task 2 - Subtask 3
    # Implement linear blended skinning

    # Pseudocode:
    #
    # For each vertex in the mesh
    #   current_pos = get_vertex(...)
    #   transformed_pos = zeros
    #   For each joint in the skeleton
    #       Get weight of joint for this vertex
    #       transformed_vertex = Compute transformed vertex position for this vertex and bone,
    #                            just like earlier for rigid_skinning
    #       transformed_pos += weight * transformed_vertex
    #   Update the transformed vertex position in the mesh (Hint: Use set_transformed_vertex)
    def linear_blend_skinning(self):
        raise NotImplementedError("TODO Task 2 Subtask 3")

    # Helper functions

    # Update the skin when joint angle changes
    def update_skin(self):
        if self.skin_mode == "rigid":
            self.rigid_skinning()
        elif self.skin_mode == "linear":
            self.linear_blend_skinning()
        else:
            raise ValueError(f"Unknown skin_mode {self.skin_mode}")

    # Create cylinder mesh for task 1 & 2
    @classmethod
    def create_cylinder_skinmesh(
        cls, skeleton: skeleton.Skeleton, skin_mode: Literal["rigid", "linear"], radius=1.0
    ) -> "SkinMesh":
        startX, endX = -2.0, 2.0
        numXSegments = 16
        numThetaBands = 16
        factor = (endX - startX) / numXSegments
        original_positions = []
        transformed_positions = []
        indices = []

        for i in range(numXSegments + 1):
            for j in range(numThetaBands):
                theta = 2 * np.pi * j / numThetaBands

                y = radius * np.sin(theta)
                z = radius * np.cos(theta)

                original_positions.append((startX, y, z))
                transformed_positions.append((startX, y, z))

                if i < numXSegments:
                    i0, i1 = i, i + 1
                    j0, j1 = j, (j + 1) % numThetaBands
                    indices.append(
                        (
                            i0 * numThetaBands + j0,
                            i0 * numThetaBands + j1,
                            i1 * numThetaBands + j1,
                        )
                    )
                    indices.append(
                        (
                            i0 * numThetaBands + j0,
                            i1 * numThetaBands + j1,
                            i1 * numThetaBands + j0,
                        )
                    )
            startX += factor
        return cls(
            np.array(original_positions),
            np.array(indices),
            np.array(transformed_positions),
            skeleton,
            skin_mode,
        )

    # Create arm mesh for task 3
    @classmethod
    def create_arm_skinmesh(
        cls, skeleton: skeleton.Skeleton, skin_mode: Literal["rigid", "linear"]
    ) -> "SkinMesh":
        original_positions = []
        transformed_positions = []
        indices = []
        for i in range(len(arm.Positions)):
            original_positions.append(arm.Positions[i])
            transformed_positions.append(arm.Positions[i])

            if i % 3 == 0:
                original_positions[i] -= -10.0
                transformed_positions[i] -= -10.0

        for i in range(len(arm.Indices)):
            indices.append(arm.Indices[i] - 1)

        return cls(
            np.array(original_positions).reshape(-1, 3),
            np.array(indices).reshape(-1, 3),
            np.array(transformed_positions).reshape(-1, 3),
            skeleton,
            skin_mode,
        )
