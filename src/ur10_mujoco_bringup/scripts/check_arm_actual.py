#!/usr/bin/env python3
"""
Send the arm to home, then to pickup_approach, then to container_center_floor,
then back. After each move, read the ACTUAL joint state and compare against
the commanded joint values. This tells us what the arm really does.
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


STEPS = [
    ("home", None),
    ("pickup_approach",
     [0.500, 0.000, 0.550, 0.000, 0.707, 0.000, 0.707]),
    ("container_center_floor",
     [-0.400, 0.000, -0.450, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_approach_again",
     [0.500, 0.000, 0.550, 0.000, 0.707, 0.000, 0.707]),
]


class ArmActualChecker(Node):
    def __init__(self):
        super().__init__("check_arm_actual")
        self.cb = ReentrantCallbackGroup()

        self.latest = None
        self.create_subscription(
            JointState, "/joint_states", self._on_js, 10,
            callback_group=self.cb,
        )

        self.ik_cli = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb
        )
        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
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

    def _on_js(self, msg):
        if len(msg.name) != 6:
            return
        d = dict(zip(msg.name, msg.position))
        try:
            self.latest = [d[j] for j in JOINT_NAMES]
        except KeyError:
            pass

    def get_joints(self):
        return self.latest

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

        future = self.ik_cli.call_async(req)
        deadline = time.time() + 4.0
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)
        if future.result() is None or future.result().error_code.val != 1:
            return None
        js = future.result().solution.joint_state
        d = dict(zip(js.name, js.position))
        return [d.get(j, 0.0) for j in JOINT_NAMES]

    def format_joints(self, j):
        if j is None:
            return "(no data)"
        return "[" + ", ".join(f"{v:+.3f}" for v in j) + "]"


def main():
    rclpy.init()
    node = ArmActualChecker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run_test():
        # Wait for joint states
        deadline = time.time() + 5.0
        while node.get_joints() is None and time.time() < deadline:
            time.sleep(0.1)

        for label, pose in STEPS:
            node.get_logger().info("=" * 70)
            node.get_logger().info(f"STEP: {label}")

            if pose is None:
                commanded = HOME_JOINTS
            else:
                commanded = node.solve_ik(pose)
                if commanded is None:
                    node.get_logger().warn("  IK FAILED")
                    continue

            node.get_logger().info(f"  COMMANDED: {node.format_joints(commanded)}")

            before = node.get_joints()
            node.get_logger().info(f"  BEFORE:    {node.format_joints(before)}")

            node.moveit2.move_to_configuration(list(commanded))
            ok = node.moveit2.wait_until_executed()
            node.get_logger().info(f"  MoveIt result: {'OK' if ok else 'FAILED'}")

            time.sleep(2.5)  # let the arm settle

            after = node.get_joints()
            node.get_logger().info(f"  AFTER:     {node.format_joints(after)}")

            if after is not None and commanded is not None:
                errors = [abs(a - c) for a, c in zip(after, commanded)]
                max_err = max(errors)
                node.get_logger().info(
                    f"  MAX JOINT ERROR: {max_err:.4f} rad "
                    f"({max_err * 57.2958:.2f} deg)"
                )
                if max_err < 0.05:
                    node.get_logger().info("  ARM REACHED COMMAND")
                else:
                    node.get_logger().warn("  ARM DID NOT REACH COMMAND")

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
