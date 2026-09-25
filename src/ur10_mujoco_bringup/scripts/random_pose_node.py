#!/usr/bin/env python3
"""
Random Pose Generator Node for UR10.

Generates random Cartesian positions and orientations within the robot's
reachable workspace and asks MoveIt2 to plan and execute motions to them.
"""

import random
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur


class RandomPoseNode(Node):
    def __init__(self):
        super().__init__("random_pose_node")

        self.declare_parameter("num_poses", 5)
        self.declare_parameter("delay_between_poses", 3.0)
        self.declare_parameter("x_range", [0.3, 0.8])
        self.declare_parameter("y_range", [-0.4, 0.4])
        self.declare_parameter("z_range", [0.2, 0.8])

        self.num_poses = self.get_parameter("num_poses").value
        self.delay = self.get_parameter("delay_between_poses").value
        self.x_range = self.get_parameter("x_range").value
        self.y_range = self.get_parameter("y_range").value
        self.z_range = self.get_parameter("z_range").value

        self.cb_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=self.cb_group,
        )

        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self.timer = self.create_timer(2.0, self.run_once)
        self.pose_count = 0

        self.get_logger().info(
            f"Random pose node ready. Will generate {self.num_poses} poses."
        )

    def run_once(self):
        if self.pose_count >= self.num_poses:
            self.get_logger().info("All poses completed. Shutting down.")
            self.timer.cancel()
            return

        self.pose_count += 1

        position = [
            random.uniform(*self.x_range),
            random.uniform(*self.y_range),
            random.uniform(*self.z_range),
        ]

        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)
        quat_xyzw = quat.tolist()

        self.get_logger().info(
            f"[Pose {self.pose_count}/{self.num_poses}] "
            f"pos=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        )

        self.moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw)
        self.moveit2.wait_until_executed()

        self.get_logger().info(f"[Pose {self.pose_count}] Executed.")


def main(args=None):
    rclpy.init(args=args)
    node = RandomPoseNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
