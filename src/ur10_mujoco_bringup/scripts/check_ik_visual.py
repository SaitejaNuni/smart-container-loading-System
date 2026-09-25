#!/usr/bin/env python3
"""
IK sequence test with CORRECTED frames and CONDITIONAL elbow constraint.

Frames:
  - pickup targets: TF world = base_link (coincident)
  - placement targets: converted from container-internal to MJCF world
    using internal_y - 0.9, then to TF world by subtracting (0.9, 0, 0.75).

Elbow constraint:
  - Applied only for targets with x > 0 (pickup side). Those targets
    clearly prefer the elbow-positive branch.
  - Skipped for container-side targets (x < 0), where the elbow-up branch
    cannot reach the deeper release point. Letting IK choose the branch
    for container side avoids the "IK FAILED" we saw on release.

The IK request is still seeded with the current joint state so that
pick_ik returns a solution close to where the arm already is.

Usage:
    ros2 run ur10_mujoco_bringup check_ik_visual.py
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    PositionIKRequest,
    Constraints,
    JointConstraint,
)
from sensor_msgs.msg import JointState

from pymoveit2 import MoveIt2


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


HOME_JOINTS = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


SEQUENCE = [
    ("pickup_approach",
     [0.500, 0.000, 0.457, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_grasp",
     [0.500, 0.000, 0.307, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_lift",
     [0.500, 0.000, 0.457, 0.000, 0.707, 0.000, 0.707]),
    ("placement_approach",
     [-0.711, -0.750, -0.293, 0.000, 0.707, 0.000, 0.707]),
    ("placement_release",
     [-0.711, -0.750, -0.443, 0.000, 0.707, 0.000, 0.707]),
]


DWELL_SEC = 1.5


class SeededSequenceChecker(Node):
    def __init__(self):
        super().__init__("check_ik_sequence")
        self.callback_group = ReentrantCallbackGroup()

        self.latest_joint_state = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            10,
            callback_group=self.callback_group,
        )

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
        self.moveit2.planner_id = "RRTstar"

        self.get_logger().info("MoveIt2 interface ready")

    def _on_joint_state(self, msg):
        if len(msg.name) != 6:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            self.latest_joint_state = [
                name_to_pos[j] for j in JOINT_NAMES
            ]
        except KeyError:
            pass

    def get_current_joints(self):
        return self.latest_joint_state

    def query_ik(self, target_pose, seed_joints):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 2

        # Seed with the current joint state so pick_ik prefers a nearby solution.
        seed_state = JointState()
        seed_state.name = list(JOINT_NAMES)
        seed_state.position = list(seed_joints)
        req.ik_request.robot_state.joint_state = seed_state

        # Apply elbow constraint ONLY for pickup-side targets (x > 0).
        if target_pose[0] > 0.0:
            jc = JointConstraint()
            jc.joint_name = "elbow_joint"
            jc.position = 1.5
            jc.tolerance_above = 1.5
            jc.tolerance_below = 1.5
            jc.weight = 1.0
            constraints = Constraints()
            constraints.joint_constraints.append(jc)
            req.ik_request.constraints = constraints
            self.get_logger().info(
                "  (elbow constraint applied: pickup side)"
            )
        else:
            self.get_logger().info(
                "  (elbow constraint skipped: container side)"
            )

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
        deadline = time.time() + 4.0
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
    node = SeededSequenceChecker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run_test():
        deadline = time.time() + 5.0
        while node.get_current_joints() is None and time.time() < deadline:
            time.sleep(0.1)

        node.get_logger().info("Moving to home...")
        node.move_to_joints_and_wait(HOME_JOINTS, "home")
        time.sleep(DWELL_SEC)

        for label, pose in SEQUENCE:
            node.get_logger().info("=" * 60)
            node.get_logger().info(f"Target: {label}")
            node.get_logger().info(f"  pose = {pose}")

            seed = node.get_current_joints()
            if seed is None:
                node.get_logger().error("No joint state — aborting")
                break
            node.get_logger().info(
                "  Seed: [" + ", ".join(f"{j:+.3f}" for j in seed) + "]"
            )

            joints = node.query_ik(pose, seed)
            if joints is None:
                node.get_logger().warn("  IK FAILED — skipping")
                continue

            joints_str = "[" + ", ".join(f"{j:+.3f}" for j in joints) + "]"
            node.get_logger().info(f"  IK solution: {joints_str}")
            node.move_to_joints_and_wait(joints, label)
            time.sleep(DWELL_SEC)

        node.get_logger().info("=" * 60)
        node.get_logger().info("Sequence finished.")

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
