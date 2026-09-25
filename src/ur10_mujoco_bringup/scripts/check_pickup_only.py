#!/usr/bin/env python3
"""
Move the arm to the pickup approach and grasp poses only. Hold each for
several seconds so you can visually verify in MuJoCo and RViz whether the
tool is actually above the pickup station.

After each move, print the FK-verified TCP position so we know exactly
where the tool ended up, independent of what the planner claims.
"""

import time
import threading

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
    ("pickup_grasp",
     [0.500, 0.000, 0.400, 0.000, 0.707, 0.000, 0.707]),
]

HOLD_SECONDS = 5.0


class PickupOnlyChecker(Node):
    def __init__(self):
        super().__init__("check_pickup_only")
        self.cb = ReentrantCallbackGroup()

        self.latest = None
        self.create_subscription(
            JointState, "/joint_states", self._on_js, 10,
            callback_group=self.cb,
        )

        self.ik_cli = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb
        )
        self.fk_cli = self.create_client(
            GetPositionFK, "/compute_fk", callback_group=self.cb
        )
        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")
        while not self.fk_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_fk...")
        self.get_logger().info("Connected to /compute_ik and /compute_fk")

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

    def compute_fk(self, joints):
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = list(joints)

        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state.joint_state = js

        future = self.fk_cli.call_async(req)
        deadline = time.time() + 4.0
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)
        if future.result() is None or future.result().error_code.val != 1:
            return None
        if len(future.result().pose_stamped) == 0:
            return None
        p = future.result().pose_stamped[0].pose
        return (p.position.x, p.position.y, p.position.z)


def main():
    rclpy.init()
    node = PickupOnlyChecker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run_test():
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

            cmd_str = "[" + ", ".join(f"{j:+.3f}" for j in commanded) + "]"
            node.get_logger().info(f"  COMMANDED: {cmd_str}")

            node.moveit2.move_to_configuration(list(commanded))
            ok = node.moveit2.wait_until_executed()
            node.get_logger().info(
                f"  MoveIt result: {'OK' if ok else 'FAILED'}"
            )

            # Let it settle
            time.sleep(2.0)

            # Read actual
            actual = node.get_joints()
            if actual is not None:
                act_str = "[" + ", ".join(f"{j:+.3f}" for j in actual) + "]"
                node.get_logger().info(f"  ACTUAL:    {act_str}")

                # FK on actual
                fk_pos = node.compute_fk(actual)
                if fk_pos is not None:
                    node.get_logger().info(
                        f"  TCP (base_link): "
                        f"x={fk_pos[0]:+.3f}  y={fk_pos[1]:+.3f}  z={fk_pos[2]:+.3f}"
                    )
                    node.get_logger().info(
                        f"  TCP (MJCF world): "
                        f"x={fk_pos[0] + 0.9:+.3f}  "
                        f"y={fk_pos[1]:+.3f}  "
                        f"z={fk_pos[2] + 0.45:+.3f}"
                    )

                if pose is not None:
                    node.get_logger().info(
                        f"  Requested TCP (base_link): "
                        f"x={pose[0]:+.3f}  y={pose[1]:+.3f}  z={pose[2]:+.3f}"
                    )
                    if fk_pos is not None:
                        dp = (
                            (fk_pos[0] - pose[0]) ** 2
                            + (fk_pos[1] - pose[1]) ** 2
                            + (fk_pos[2] - pose[2]) ** 2
                        ) ** 0.5
                        node.get_logger().info(
                            f"  POSITION ERROR: {dp * 1000:.2f} mm"
                        )

            node.get_logger().info(
                f"  Holding for {HOLD_SECONDS} s — inspect MuJoCo now"
            )
            time.sleep(HOLD_SECONDS)

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
