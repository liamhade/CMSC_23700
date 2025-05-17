import polyscope as ps
import polyscope.imgui as psim
import numpy as np

from skin import SkinMesh
from skeleton import Skeleton
from joint import Joint
import cube


class Task1:
    def __init__(self):
        self.skeleton = Skeleton()

        self.joint1 = Joint(None, (-2, 0, 0), (0, 1, 0), 1.8, "UpperArm")
        self.joint2 = Joint(self.joint1, (2, 0, 0), (0, 0, 1), 1.8, "Forearm")

        self.skeleton.add_joint(self.joint1)
        self.skeleton.add_joint(self.joint2)

        self.skin_mesh = SkinMesh.create_cylinder_skinmesh(
            self.skeleton, "rigid", radius=0.5
        )

    def render(self):
        positions, indices = (
            np.array(cube.Positions).reshape((-1, 3)),
            np.array(cube.Indices).reshape((-1, 3)),
        )

        angle_upper_arm, angle_forearm = 0, 0
        init = 1

        def callback():
            nonlocal angle_upper_arm, angle_forearm, init
            changed1, angle_upper_arm = psim.SliderInt(  # type: ignore
                "Upper Arm Angle", angle_upper_arm, v_min=0, v_max=360
            )
            if changed1:
                self.set_joint_angle(0, angle_upper_arm)
            changed2, angle_forearm = psim.SliderInt(  # type: ignore
                "Forearm Angle", angle_forearm, v_min=0, v_max=360
            )
            if changed2:
                self.set_joint_angle(1, angle_forearm)

            if changed1 or changed2 or init:
                ps_mesh = ps.register_surface_mesh(
                    "cylinder",
                    np.array(self.skin_mesh.transformed_positions).reshape((-1, 3)),
                    np.array(self.skin_mesh.indices).reshape((-1, 3)),
                )
                ps_mesh.add_scalar_quantity(
                    "weights",
                    np.array(self.skin_mesh.weights).reshape((-1, 2))[:, 0],
                    enabled=True,
                )
                if init:
                    ps.set_automatically_compute_scene_extents(False)
                    init = 0

                transformed_positions = positions.copy()
                for i, p in enumerate(transformed_positions):
                    transformed_positions[i] = (
                        self.joint1.compute_model_matrix() @ [*p, 1]
                    )[:-1]
                ps.register_surface_mesh("Upper Arm", transformed_positions, indices)

                transformed_positions = positions.copy()
                for i, p in enumerate(transformed_positions):
                    transformed_positions[i] = (
                        self.joint2.compute_model_matrix() @ [*p, 1]
                    )[:-1]
                ps.register_surface_mesh("Forearm", transformed_positions, indices)

        ps.init()
        ps.set_user_callback(callback)
        ps.show()

    def set_joint_angle(self, joint_idx: int, joint_angle: float):
        self.skeleton.get_joint(joint_idx).joint_angle = joint_angle
        self.skin_mesh.update_skin()


class Task2:
    def __init__(self):
        self.skeleton = Skeleton()

        self.joint1 = Joint(None, (-2, 0, 0), (0, 1, 0), 1.8, "UpperArm")
        self.joint2 = Joint(self.joint1, (2, 0, 0), (0, 0, 1), 1.8, "Forearm")

        self.skeleton.add_joint(self.joint1)
        self.skeleton.add_joint(self.joint2)

        self.skin_mesh = SkinMesh.create_cylinder_skinmesh(
            self.skeleton, "linear", radius=0.5
        )

    def render(self):
        positions, indices = (
            np.array(cube.Positions).reshape((-1, 3)),
            np.array(cube.Indices).reshape((-1, 3)),
        )

        angle_upper_arm, angle_forearm = 0, 0
        init = 1

        def callback():
            nonlocal angle_upper_arm, angle_forearm, init
            changed1, angle_upper_arm = psim.SliderInt(  # type: ignore
                "Upper Arm Angle", angle_upper_arm, v_min=0, v_max=360
            )
            if changed1:
                self.set_joint_angle(0, angle_upper_arm)
            changed2, angle_forearm = psim.SliderInt(  # type: ignore
                "Forearm Angle", angle_forearm, v_min=0, v_max=360
            )
            if changed2:
                self.set_joint_angle(1, angle_forearm)

            if changed1 or changed2 or init:
                ps_mesh = ps.register_surface_mesh(
                    "cylinder",
                    np.array(self.skin_mesh.transformed_positions).reshape((-1, 3)),
                    np.array(self.skin_mesh.indices).reshape((-1, 3)),
                )
                ps_mesh.add_scalar_quantity(
                    "weights",
                    np.array(self.skin_mesh.weights).reshape((-1, 2))[:, 0],
                    enabled=True,
                )
                if init:
                    ps.set_automatically_compute_scene_extents(False)
                    init = 0

                transformed_positions = positions.copy()
                for i, p in enumerate(transformed_positions):
                    transformed_positions[i] = (
                        self.joint1.compute_model_matrix() @ [*p, 1]
                    )[:-1]
                ps.register_surface_mesh("Upper Arm", transformed_positions, indices)

                transformed_positions = positions.copy()
                for i, p in enumerate(transformed_positions):
                    transformed_positions[i] = (
                        self.joint2.compute_model_matrix() @ [*p, 1]
                    )[:-1]
                ps.register_surface_mesh("Forearm", transformed_positions, indices)

        ps.init()
        ps.set_user_callback(callback)
        ps.show()

    def set_joint_angle(self, joint_idx: int, joint_angle: float):
        self.skeleton.get_joint(joint_idx).joint_angle = joint_angle
        self.skin_mesh.update_skin()
