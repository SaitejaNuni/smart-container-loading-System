#!/usr/bin/env python3
"""
Diagnose the two problems:
  A. Does the tool tip reach the top of the package at pickup?
  B. Where does the package attach after weld activation?

Uses tf2_echo-like reads for the tool pose, then reads the package body
pose from MuJoCo's set_free_joint_state callback path... but since we do
not have direct MuJoCo access here, we use FK from the joint state after
each move plus the known world offsets.

Reports:
  target_z              : where the tool SHOULD be
  achieved_site_z       : where the tool tip IS (via /compute_fk)
  package_centre_after  : where the package body is after weld
"""

import re
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from thesis_msgs.srv import GetNextPackage

from pymoveit2 import MoveIt2
from ssik.prebuilt import ur10e_ik


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

HOME2_Q = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
TOOL_DOWN_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
PICKUP_TOP_TF_Z = 0.30
PICKUP_TOOL_OFFSET = 0.0

T_DH_FROM_URDF = np.array([
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0, -1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
])


def pose_to_matrix(pos, quat_xyzw):
    x, y, z, w = quat_xyzw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wraparound_distance(q, seed):
    return float(np.linalg.norm(wrap_angle(np.asarray(q) - np.asarray(seed))))


class Diagnostics(Node):
    def __init__(self):
        super().__init__("diagnose_weld")
        self.cb = ReentrantCallbackGroup()

        self.latest_js = None
        self.create_subscription(
            JointState, "/joint_states", self._on_js, 10,
            callback_group=self.cb)

        self.active_idx_pub = self.create_publisher(
            Int32, "/active_package_index", 10)
        self.gripper_pub = self.create_publisher(
            Bool, "/gripper_command", 10)
        self.spawn_cli = self.create_client(
            Trigger, "/spawn_package", callback_group=self.cb)
        self.gen_cli = self.create_client(
            GetNextPackage, "/get_next_package", callback_group=self.cb)
        self.fk_cli = self.create_client(
            GetPositionFK, "/compute_fk", callback_group=self.cb)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            callback_group=self.cb,
        )
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3
        self.moveit2.planner_id = "ur_manipulator[RRTstar]"

    def _on_js(self, msg):
        self.latest_js = msg

    def get_joints(self):
        if self.latest_js is None:
            return None
        d = dict(zip(self.latest_js.name, self.latest_js.position))
        try:
            return [d[j] for j in JOINT_NAMES]
        except KeyError:
            return None

    def fk_tool0(self, joints):
        """Use /compute_fk to get tool0 pose for given joints."""
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in joints]
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state.joint_state = js
        if not self.fk_cli.wait_for_service(timeout_sec=5.0):
            return None
        fut = self.fk_cli.call_async(req)
        deadline = time.time() + 4.0
        while not fut.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return None
        if len(res.pose_stamped) == 0:
            return None
        p = res.pose_stamped[0].pose.position
        return (p.x, p.y, p.z)

    def get_next_package(self):
        if not self.gen_cli.wait_for_service(timeout_sec=5.0):
            return None
        fut = self.gen_cli.call_async(GetNextPackage.Request())
        deadline = time.time() + 5.0
        while not fut.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)
        res = fut.result()
        if res is None or not res.has_next:
            return None
        return res.package

    def spawn_package(self, pkg):
        m = re.search(r"(\d+)$", pkg.id)
        if m is None:
            return False
        idx = int(m.group(1))
        msg = Int32()
        msg.data = idx
        self.active_idx_pub.publish(msg)
        time.sleep(0.3)
        if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
            return False
        fut = self.spawn_cli.call_async(Trigger.Request())
        deadline = time.time() + 5.0
        while not fut.done():
            if time.time() > deadline:
                return False
            time.sleep(0.02)
        res = fut.result()
        return res is not None and res.success

    def move_joints(self, joints, label):
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[{label}] moving to joints")
        self.moveit2.move_to_configuration(list(joints))
        ok = self.moveit2.wait_until_executed()
        self.get_logger().info(f"[{label}] result: {'OK' if ok else 'FAILED'}")
        return ok

    def analytical_ik(self, target_pos_urdf, target_quat_xyzw, seed):
        T_urdf = pose_to_matrix(target_pos_urdf, target_quat_xyzw)
        T_dh = T_DH_FROM_URDF @ T_urdf
        sols = ur10e_ik.solve(T_dh, max_solutions=8,
                              q_seed=np.asarray(seed),
                              seed_metric="wrap_linf",
                              respect_limits=True)
        if not sols:
            return None
        best = None
        best_d = float("inf")
        for s in sols:
            q = np.asarray(s.q)
            d = wraparound_distance(q, seed)
            if d < best_d:
                best_d = d
                best = q
        return best


def main():
    rclpy.init()
    node = Diagnostics()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run():
        time.sleep(2.0)
        pkg = node.get_next_package()
        if pkg is None:
            return
        node.get_logger().info(f"pkg {pkg.id}  H={pkg.height:.4f}")
        if not node.spawn_package(pkg):
            return
        time.sleep(2.5)
        node.move_joints(HOME2_Q, "home2")
        time.sleep(1.5)

        # --- DIAGNOSTIC A: tool tip at pickup ---
        target_z = PICKUP_TOP_TF_Z + pkg.height + PICKUP_TOOL_OFFSET
        target_pos = [0.5, 0.0, target_z]
        node.get_logger().info("=" * 60)
        node.get_logger().info("DIAGNOSTIC A: tool tip at pickup")
        node.get_logger().info(f"  pkg.height    = {pkg.height:.4f}")
        node.get_logger().info(f"  pickup_top_z  = {PICKUP_TOP_TF_Z:.4f}")
        node.get_logger().info(f"  pkg center z  = "
                               f"{PICKUP_TOP_TF_Z + pkg.height/2:.4f}")
        node.get_logger().info(f"  pkg top z     = "
                               f"{PICKUP_TOP_TF_Z + pkg.height:.4f}")
        node.get_logger().info(f"  target_z      = {target_z:.4f}")

        solved = node.analytical_ik(target_pos, TOOL_DOWN_QUAT_XYZW, HOME2_Q)
        if solved is None:
            node.get_logger().error("IK failed")
            return
        node.get_logger().info(f"  chosen joints = "
                               f"{[f'{v:+.3f}' for v in solved]}")
        node.move_joints(solved, "pickup")
        time.sleep(1.5)

        # Read achieved tool0 pose via FK
        actual_joints = node.get_joints()
        if actual_joints is not None:
            tool0_pos = node.fk_tool0(actual_joints)
            if tool0_pos is not None:
                node.get_logger().info(f"  ACHIEVED tool0 in TF world:")
                node.get_logger().info(
                    f"    ({tool0_pos[0]:+.4f}, "
                    f"{tool0_pos[1]:+.4f}, "
                    f"{tool0_pos[2]:+.4f})")
                node.get_logger().info(
                    f"  TARGET tool0 in TF world:")
                node.get_logger().info(
                    f"    ({target_pos[0]:+.4f}, "
                    f"{target_pos[1]:+.4f}, "
                    f"{target_pos[2]:+.4f})")
                node.get_logger().info(
                    f"  Z error = "
                    f"{(tool0_pos[2] - target_pos[2]) * 1000:.2f} mm "
                    f"(positive = tool is ABOVE target)")

        # --- DIAGNOSTIC B: package position before and after weld ---
        node.get_logger().info("=" * 60)
        node.get_logger().info("DIAGNOSTIC B: package position after weld")
        node.get_logger().info(
            "  Watch MuJoCo: where is the package relative to the "
            "wrist cylinder and relative to the flange tip?")
        node.get_logger().info("  Activating weld in 3 seconds...")
        time.sleep(3.0)

        msg = Bool()
        msg.data = True
        node.gripper_pub.publish(msg)
        time.sleep(2.0)

        node.get_logger().info("  Weld active. Now open a NEW TERMINAL")
        node.get_logger().info("  and run:")
        node.get_logger().info("")
        node.get_logger().info(
            "    ros2 topic echo /mujoco_ros2_control_node/"
            "free_joint_states --once")
        node.get_logger().info("")
        node.get_logger().info(
            "  That prints the world pose of every package body.")
        node.get_logger().info(
            "  Compare the package_XX entry with the wrist_3_link pose.")
        node.get_logger().info(
            "  Send both numbers and I will tell you the exact weld")
        node.get_logger().info(
            "  offset that puts the package at the flange.")
        time.sleep(30.0)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
