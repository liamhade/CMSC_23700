# The joint class represents a joint and its attached bone.
# The joint is attached at 'position' relative to its parent and rotates around 'joint_axis'
from typing import Optional, Union, Sequence
import numpy as np
from transforms3d.affines import compose
from transforms3d.axangles import axangle2mat

"""
TODO -- Task 1: complete the functions
get_local_matrix, get_world_matrix, compute_binding_matrix
"""

class Joint:
    def __init__(
        self,
        parent: Optional["Joint"],
        position: tuple[float, float, float],
        joint_axis: tuple[float, float, float],
        length: float,
        name: str,
    ):
        self.parent = parent  # if parent is None, that means this Joint is the root
        self.position = position
        self.joint_axis = joint_axis
        self.joint_angle = 0.0  # this is in degrees
        self.name = name
        self.length = length

        # The binding matrix stores the orientation of the bone at the time it was attached
        # to the skin. This matrix is None until the bone is attached to the skin
        self.binding_matrix: Optional[np.ndarray] = None

    # TODO: Task 1 - Subtask 1
    #
    # Returns the local transform of the current joint
    # This matrix should rotate by 'self.joint_angle' around 'self.joint_axis'
    # and then translate by 'self.position'
    # Hint: self.translation_matrix() and self.rotation_matrix() are helpful
    # This should be a 4x4 matrix (an array of shape (4,4)).
    def get_local_matrix(self) -> np.ndarray:
        Rm = self.rotation_matrix(self.joint_angle, *self.joint_axis)
        Tm = self.translation_matrix(*self.position)
        return Tm @ Rm


    # TODO: Task 1 - Subtask 1
    #
    # Returns the world transform of the current joint.
    # This is simply the transform of the parent joint (if any) multiplied by this joint's local transform
    # Hint: use the get_local_matrix function you implemented above.
    # This should be a 4x4 matrix (an array of shape (4,4))
    def get_world_matrix(self) -> np.ndarray:
        if not self.parent:
            return self.get_local_matrix()
        else:
            return self.parent.get_world_matrix() @ self.get_local_matrix()

    # TODO: Task 1 - Subtask 1
    #
    # Compute the binding transform matrix of the joint.
    # Hint: The binding matrix transforms points from world space to the local space of the joint
    # Use get_world_matrix and a matrix inverse.
    # This should be a 4x4 matrix (an array of shape (4,4))
    # Set self.binding_matrix to the result (no need to return the result)
    def compute_binding_matrix(self):
        self.binding_matrix = np.linalg.inv(self.get_world_matrix())

    # Helper functions

    def get_binding_matrix(self) -> Optional[np.ndarray]:
        return self.binding_matrix

    # Returns endpoints of joint in world space: (v0, v1)
    # Used to compute distance to line segment
    def get_joint_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        posemat = self.get_world_matrix()
        return (
            (posemat @ [0, 0, 0, 1])[:-1],
            (posemat @ [self.length, 0, 0, 1])[:-1],
        )

    # Computes model matrix for rendering
    def compute_model_matrix(self) -> np.ndarray:
        posemat = self.get_world_matrix()
        sMatrix = compose(np.zeros(3), np.identity(3), np.array([self.length, 0.2, 0.2]))
        tMatrix = compose(np.array([self.length / 2, 0, 0]), np.identity(3), np.ones(3))
        return posemat @ tMatrix @ sMatrix

    # 4x4 matrix that translates by (x, y, z)
    @staticmethod
    def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
        return compose(np.array([x, y, z]), np.identity(3), np.ones(3))

    # 4x4 matrix that rotates by degrees theta around the axis (x, y, z)
    @staticmethod
    def rotation_matrix(theta: float, x: float, y: float, z: float) -> np.ndarray:
        rotation3x3 = axangle2mat(np.array([x, y, z]), np.radians(theta))
        return compose(np.zeros(3), rotation3x3, np.ones(3))

if __name__ == "__main__":
    parent = None
    position = (1.5,2.54,3.3)
    joint_axis = (1,2,3)
    length = 13.1
    name = "knee"

    j = Joint(parent, position, joint_axis, length, name)
    j.compute_binding_matrix()
    Lm = j.get_local_matrix()
    Wm = j.get_world_matrix()
    Bm = j.get_binding_matrix()

    print(Lm)
    print(Wm)
    print(Bm)