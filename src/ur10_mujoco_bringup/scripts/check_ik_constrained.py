#!/usr/bin/env python3
"""
Test IK with an explicit joint constraint on elbow_joint, plus RRTstar
as the trajectory planner.

For each target pose:
  1. Query /compute_ik with a constraint forcing elbow_joint in [0, 3.0]
  2. Print the resulting joints
  3. Move the arm to those joints (MoveIt plans the path with RRTstar)

If both pickup and placement targets reach the same branch, the weird
motion disappears.

Usage:
    ros2 run ur10_mujoco_bringup check_ik_constrained.py
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint

from pymoveit2 import MoveIt2


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# Elbow constraint: force elbow positive (elbow-up branch)
ELBOW_CENTER = 1.5
ELBOW_TOLERANCE = 1.5  # range becomes [0.0, 3.0]


SEQUENCE = [
    ("pickup_approach",
     [0.500, 0.000, 0.457, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_grasp",
     [0.500, 0.000, 0.307, 0.000, 0.707, 0.000, 0.707]),
    ("placement_approach",
     [-0.711, 0.150, -0.293, 0.000, 0.707, 0.000, 0.707]),
    ("placement_release",
     [-0.711, 0.150, -0.443, 0.000, 0.707, 0.000, 0.707]),
]


DWELL_SEC = 2.0


class ConstrainedIKChecker(Node):
    def __init__(self):
        super().__init__("check_ik_constrained")
        self.callback_group = ReentrantCallbackGroup()

        self.cli = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=self.callback_group,
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")
        self.get_logger().info("Connected to /compute_ik")

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            callback_group=self.callback_group,
        )
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5

        # Use RRTstar for smoother paths
        self.moveit2.planner_id = "RRTstar"
        self.get_logger().info("Planner: RRTstar")

        self.get_logger().info("MoveIt2 interface ready")

    def query_ik_constrained(self, target_pose):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 2

        # Elbow constraint
        jc = JointConstraint()
        jc.joint_name = "elbow_joint"
        jc.position = ELBOW_CENTER
        jc.tolerance_above = ELBOW_TOLERANCE
        jc.tolerance_below = ELBOW_TOLERANCE
        jc.weight = 1.0

        constraints = Constraints()
        constraints.joint_constraints.append(jc)
        req.ik_request.constraints = constraints

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose.position.x = target_pose[0]
        pose_stamped.pose.position.y = target_pose[1]
        pose_stamped.pose.position.z = target_pose[2]
        pose_stamped.pose.orientation.x = target_pose[3]
        pose_stamped.pose.orientation.y = target_pose[4]
        pose_stamped.pose.orientation.z = target_pose[5]
        pose_stamped.pose.orientation.w = target_pose[6]
        req.ik_request.pose_stamped = pose_stamped

        future = self.cli.call_async(req)
        deadline = time.time() + 3.0
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)

        if future.result() is None:
            return None
        if future.result().error_code.val != 1:
            return None

        js = future.result().solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        return [name_to_pos.get(j, 0.0) for j in JOINT_NAMES]

    def move_to_joints_and_wait(self, joints, label):
        self.get_logger().info(f"  [{label}] Move to configuration...")
        self.moveit2.move_to_configuration(list(joints))
        ok = self.moveit2.wait_until_executed()
        self.get_logger().info(
            f"  [{label}] Result: {'OK' if ok else 'FAILED'}"
        )
        return ok


def main():
    rclpy.init()
    node = ConstrainedIKChecker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run_test():
        for label, pose in SEQUENCE:
            node.get_logger().info("=" * 60)
            node.get_logger().info(f"Target: {label}")
            node.get_logger().info(f"  pose = {pose}")

            joints = node.query_ik_constrained(pose)
            if joints is None:
                node.get_logger().error(
                    "  IK FAILED with elbow constraint in [0, 3]. "
                    "This target is not reachable on the elbow-up branch."
                )
                continue

            joints_str = "[" + ", ".join(f"{j:+.3f}" for j in joints) + "]"
            node.get_logger().info(f"  IK solution: {joints_str}")
            node.move_to_joints_and_wait(joints, label)
            time.sleep(DWELL_SEC)

        node.get_logger().info("=" * 60)
        node.get_logger().info("Test finished.")

    t = threading.Thread(target=run_test, daemon=True)
    t.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        t.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
