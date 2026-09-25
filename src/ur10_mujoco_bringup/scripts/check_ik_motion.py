#!/usr/bin/env python3
"""
Move the arm through the joint solutions that pick_ik returns for each
target. This is a visual test: watch MuJoCo to see whether the arm moves
smoothly or flips between branches.
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest

from pymoveit2 import MoveIt2


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


TARGETS = [
    ("pickup_approach",
     [0.500, 0.000, 0.550, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_grasp",
     [0.500, 0.000, 0.400, 0.000, 0.707, 0.000, 0.707]),
    ("container_center_floor",
     [-0.400, 0.000, -0.450, 0.000, 0.707, 0.000, 0.707]),
    ("container_near_corner_floor",
     [0.100, 0.900, -0.450, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_approach_return",
     [0.500, 0.000, 0.550, 0.000, 0.707, 0.000, 0.707]),
]


class MotionChecker(Node):
    def __init__(self):
        super().__init__("check_ik_motion")
        self.cb = ReentrantCallbackGroup()

        self.cli = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb
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
            callback_group=self.cb,
        )
        self.moveit2.max_velocity = 0.5
        self.moveit2.max_acceleration = 0.5
        self.moveit2.planner_id = "RRTstar"

    def solve_ik(self, pose):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 2

        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.pose.position.x = pose[0]
        ps.pose.position.y = pose[1]
        ps.pose.position.z = pose[2]
        ps.pose.orientation.x = pose[3]
        ps.pose.orientation.y = pose[4]
        ps.pose.orientation.z = pose[5]
        ps.pose.orientation.w = pose[6]
        req.ik_request.pose_stamped = ps

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


def main():
    rclpy.init()
    node = MotionChecker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run_test():
        time.sleep(2.0)
        for label, pose in TARGETS:
            node.get_logger().info("=" * 60)
            node.get_logger().info(f"Target: {label}")
            joints = node.solve_ik(pose)
            if joints is None:
                node.get_logger().warn("  IK FAILED")
                continue

            joints_str = "[" + ", ".join(f"{j:+.3f}" for j in joints) + "]"
            node.get_logger().info(f"  IK solution: {joints_str}")
            node.get_logger().info("  Moving...")

            node.moveit2.move_to_configuration(joints)
            ok = node.moveit2.wait_until_executed()
            node.get_logger().info(
                f"  Result: {'OK' if ok else 'FAILED'}"
            )
            time.sleep(2.0)

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
