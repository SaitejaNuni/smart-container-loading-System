#!/usr/bin/env python3
"""
Direct pickup test with the CORRECTED top-down quaternion [1, 0, 0, 0].

Previously we sent [0, 0.707, 0, 0.707] which pointed the tool sideways.
Now the tool points down, so the package will hang below the wrist.
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import PositionIKRequest

from pymoveit2 import MoveIt2


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

HOME = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# CORRECTED: tool pointing down
TOOL_QUAT = [1.0, 0.0, 0.0, 0.0]

STEPS = [
    ("home", None),
    ("pickup_approach", [0.500, 0.000, 0.680] + TOOL_QUAT),
    ("pickup_grasp",    [0.500, 0.000, 0.530] + TOOL_QUAT),
]


def wait_future(future, timeout):
    deadline = time.time() + timeout
    while not future.done():
        if time.time() > deadline:
            return False
        time.sleep(0.05)
    return True


class PickupTester(Node):
    def __init__(self):
        super().__init__("test_pickup_direct")
        self.cb = ReentrantCallbackGroup()
        self.latest = None
        self.create_subscription(
            JointState, "/joint_states", self._js, 10, callback_group=self.cb,
        )
        self.ik_cli = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb)
        self.fk_cli = self.create_client(
            GetPositionFK, "/compute_fk", callback_group=self.cb)
        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")
        while not self.fk_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_fk...")

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            callback_group=self.cb,
        )
        self.moveit2.max_velocity = 0.4
        self.moveit2.max_acceleration = 0.4
        self.moveit2.planner_id = "RRTstar"

    def _js(self, msg):
        d = dict(zip(msg.name, msg.position))
        try:
            self.latest = [d[j] for j in JOINT_NAMES]
        except KeyError:
            pass

    def solve_ik(self, pose):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = True
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
        fut = self.ik_cli.call_async(req)
        if not wait_future(fut, 5.0):
            return None
        if fut.result() is None or fut.result().error_code.val != 1:
            return None
        js = fut.result().solution.joint_state
        d = dict(zip(js.name, js.position))
        return [d.get(j, 0.0) for j in JOINT_NAMES]

    def fk(self, joints):
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = list(joints)
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state.joint_state = js
        fut = self.fk_cli.call_async(req)
        if not wait_future(fut, 5.0):
            return None
        if fut.result() is None or len(fut.result().pose_stamped) == 0:
            return None
        p = fut.result().pose_stamped[0].pose.position
        return (p.x, p.y, p.z)


def main():
    rclpy.init()
    node = PickupTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run():
        deadline = time.time() + 5.0
        while node.latest is None and time.time() < deadline:
            time.sleep(0.1)

        for label, pose in STEPS:
            node.get_logger().info("=" * 60)
            node.get_logger().info(f"STEP: {label}")

            cmd = HOME if pose is None else node.solve_ik(pose)
            if cmd is None:
                node.get_logger().error("IK FAILED")
                continue

            node.get_logger().info(
                "  COMMANDED: [" + ", ".join(f"{j:+.3f}" for j in cmd) + "]"
            )
            node.moveit2.move_to_configuration(list(cmd))
            ok = node.moveit2.wait_until_executed()
            node.get_logger().info(f"  MoveIt: {'OK' if ok else 'FAILED'}")
            time.sleep(2.0)

            actual = node.latest
            node.get_logger().info(
                "  ACTUAL:    [" + ", ".join(f"{j:+.3f}" for j in actual) + "]"
            )
            fk = node.fk(actual)
            if fk is not None:
                node.get_logger().info(
                    f"  TCP base_link: ({fk[0]:+.3f}, {fk[1]:+.3f}, {fk[2]:+.3f})"
                )
                node.get_logger().info(
                    f"  TCP MJCF:      ({fk[0] + 0.9:+.3f}, {fk[1]:+.3f}, {fk[2] + 0.45:+.3f})"
                )
                node.get_logger().info(
                    "  (pickup station top in MJCF is at 1.4, 0, 0.75)"
                )

            time.sleep(4.0)

        node.get_logger().info("Done.")

    t = threading.Thread(target=run, daemon=True)
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
