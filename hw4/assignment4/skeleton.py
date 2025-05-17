from joint import Joint


# Skeleton class contains all the bones in a list,
# the hierarchy is established through the bones themselves.
class Skeleton:
    def __init__(self):
        self.joints: list[Joint] = []

    # Add a joint to joints
    def add_joint(self, joint: Joint):
        self.joints.append(joint)

    # Given an index, returns the joint
    def get_joint(self, index: int) -> Joint:
        return self.joints[index]

    # Return number of joints in skeleton
    def get_num_joints(self) -> int:
        return len(self.joints)

    # computes binding matrix for all joints
    def compute_binding_matrices(self):
        for joint in self.joints:
            joint.compute_binding_matrix()
