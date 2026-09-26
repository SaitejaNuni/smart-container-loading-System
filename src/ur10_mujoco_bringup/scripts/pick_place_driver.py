#!/usr/bin/env python3
"""
Multi-package pick-and-place driver with metrics logging.

Runs the full sequence N times, where N is set by the `num_packages`
parameter (default 1). Writes a CSV row per cycle to:
    ~/thesis_mujoco_ws/placement_metrics.csv

Run:
    ros2 run ur10_mujoco_bringup pick_place_driver.py
    ros2 run ur10_mujoco_bringup pick_place_driver.py --ros-args -p num_packages:=3
"""

import json
import os
import re
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, Float64MultiArray, Int32, String
from std_srvs.srv import Trigger
from thesis_msgs.srv import GetNextPackage, PlanPlacement

from pymoveit2 import MoveIt2
from ssik.prebuilt import ur10e_ik


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

PICKUP_HEIGHT_FACTOR = 0.6
WELD_QUAT_WXYZ = (0.7071, 0.7071, 0.0, 0.0)
RELPOSE_SETTLE = 0.15

# =============================================================================


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

HOME_Q = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
HOME2_Q = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

TOOL_DOWN_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]

PICKUP_TOP_TF_Z = 0.30
CONTAINER_INTERNAL_TO_TF = np.array([-0.9, -0.9, -0.45])

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


class PickPlaceDriver(Node):
    def __init__(self):
        super().__init__("pick_place_driver")
        self.cb = ReentrantCallbackGroup()

        self.declare_parameter("num_packages", 1)
        self.num_packages = self.get_parameter("num_packages").value

        self.active_idx_pub = self.create_publisher(
            Int32, "/active_package_index", 10)
        self.gripper_pub = self.create_publisher(
            Bool, "/gripper_command", 10)
        self.relpose_pub = self.create_publisher(
            Float64MultiArray, "/weld_plugin/set_relpose", 10)
        self.package_placed_pub = self.create_publisher(
            String, "/package_placed", 10)

        self.spawn_cli = self.create_client(
            Trigger, "/spawn_package", callback_group=self.cb)
        self.gen_cli = self.create_client(
            GetNextPackage, "/get_next_package", callback_group=self.cb)
        self.plan_cli = self.create_client(
            PlanPlacement, "/plan_placement", callback_group=self.cb)

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
        self.moveit2.planner_id = "ur_manipulator[RRTstar]"

        # ---- Metrics CSV ----
        self.metrics_path = os.path.expanduser(
            "~/thesis_mujoco_ws/placement_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write(
                    "cycle,pkg_id,"
                    "target_container_x,target_container_y,target_container_z,"
                    "tool_target_x,tool_target_y,tool_target_z,"
                    "plan_time_s,ik_time_s,carry_time_s,settle_time_s,"
                    "cycle_time_s\n"
                )
        self.get_logger().info(f"Metrics log: {self.metrics_path}")

        # Tracked across the cycle for metric write at the end
        self._t_weld_on = 0.0

        self.get_logger().info(
            f"num_packages = {self.num_packages}")

    # ------------------------------------------------------------------

    def get_next_package(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("[gen] requesting next package")
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
        pkg = res.package
        self.get_logger().info(
            f"[gen] got {pkg.id}  H={pkg.height:.4f}")
        return pkg

    def spawn_package(self, pkg):
        m = re.search(r"(\d+)$", pkg.id)
        if m is None:
            return False
        idx = int(m.group(1))
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[spawn] requesting index {idx}")
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
        result = fut.result()
        if result is None or not result.success:
            return False
        self.get_logger().info(f"[spawn] success: {result.message}")
        return True

    def request_placement(self, pkg):
        self.get_logger().info("=" * 60)
        self.get_logger().info("[plan] requesting placement target")
        if not self.plan_cli.wait_for_service(timeout_sec=5.0):
            return None
        req = PlanPlacement.Request()
        req.package = pkg
        fut = self.plan_cli.call_async(req)
        deadline = time.time() + 15.0
        while not fut.done():
            if time.time() > deadline:
                return None
            time.sleep(0.02)
        res = fut.result()
        if res is None or not res.success:
            return None
        self.get_logger().info(
            f"[plan] internal: ({res.target.pose.position.x:+.3f}, "
            f"{res.target.pose.position.y:+.3f}, "
            f"{res.target.pose.position.z:+.3f})")
        return res.target

    def publish_relpose(self, pkg_index, height):
        offset_y = height * PICKUP_HEIGHT_FACTOR
        msg = Float64MultiArray()
        msg.data = [
            float(pkg_index),
            0.0,
            float(offset_y),
            0.0,
            float(WELD_QUAT_WXYZ[0]),
            float(WELD_QUAT_WXYZ[1]),
            float(WELD_QUAT_WXYZ[2]),
            float(WELD_QUAT_WXYZ[3]),
        ]
        self.relpose_pub.publish(msg)
        self.get_logger().info(
            f"[relpose] pkg={pkg_index}  offset_y={offset_y:.4f}")
        time.sleep(RELPOSE_SETTLE)

    def publish_placement(self, target, pkg):
        """Notify container_state that this package has been physically placed."""
        msg = String()
        msg.data = json.dumps({
            "name": pkg.id,
            "position_m": [
                float(target.pose.position.x),
                float(target.pose.position.y),
                float(target.pose.position.z),
            ],
            "orientation_quat_wxyz": [
                float(target.pose.orientation.w),
                float(target.pose.orientation.x),
                float(target.pose.orientation.y),
                float(target.pose.orientation.z),
            ],
            "dimensions_m": [
                float(target.dim_x),
                float(target.dim_y),
                float(target.dim_z),
            ],
            "mass_kg": float(pkg.mass),
            "event": "placed",
        })
        self.package_placed_pub.publish(msg)
        self.get_logger().info(
            f"[placed] {pkg.id}  center=("
            f"{target.pose.position.x:.3f}, "
            f"{target.pose.position.y:.3f}, "
            f"{target.pose.position.z:.3f})  "
            f"dims=({target.dim_x:.3f}, "
            f"{target.dim_y:.3f}, {target.dim_z:.3f})")

    def set_weld(self, active):
        self.get_logger().info(
            f"[weld] {'ON' if active else 'OFF'}")
        msg = Bool()
        msg.data = active
        self.gripper_pub.publish(msg)
        time.sleep(1.5)

    def move_joints(self, joints, label):
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[{label}] moving to joints")
        self.get_logger().info(f"  {[f'{j:+.3f}' for j in joints]}")
        self.moveit2.move_to_configuration(list(joints))
        ok = self.moveit2.wait_until_executed()
        self.get_logger().info(
            f"[{label}] result: {'OK' if ok else 'FAILED'}")
        return ok

    def analytical_ik(self, target_pos_urdf, target_quat_xyzw, seed):
        T_urdf = pose_to_matrix(target_pos_urdf, target_quat_xyzw)
        T_dh = T_DH_FROM_URDF @ T_urdf

        sols = ur10e_ik.solve(
            T_dh,
            max_solutions=8,
            q_seed=np.asarray(seed),
            seed_metric="wrap_linf",
            respect_limits=True,
        )
        if not sols:
            return None

        self.get_logger().info(f"  ssik returned {len(sols)} solutions")
        best = None
        best_d = float("inf")
        for i, s in enumerate(sols):
            q = np.asarray(s.q)
            d = wraparound_distance(q, seed)
            self.get_logger().info(
                f"    [{i}] {[f'{v:+.3f}' for v in q]}  d={d:.3f}")
            if d < best_d:
                best_d = d
                best = q
        return best

    # ------------------------------------------------------------------

    def run_one_cycle(self, cycle_num):
        t_cycle_start = time.time()

        self.get_logger().info("")
        self.get_logger().info("#" * 60)
        self.get_logger().info(
            f"# CYCLE {cycle_num}/{self.num_packages}")
        self.get_logger().info("#" * 60)

        # 0. Recover to HOME in case the previous cycle left the arm off-target.
        #    Idempotent: if the arm is already at HOME, MoveIt plans a trivial
        #    trajectory. If the previous cycle aborted mid-move, this pulls the
        #    arm back to a known state before starting a new cycle.
        if not self.move_joints(HOME_Q, "0-recover-home"):
            self.get_logger().warn(
                "0-recover-home failed; the arm may be in an unusual state")
        time.sleep(0.5)

        # 1. Get + spawn
        pkg = self.get_next_package()
        if pkg is None:
            return False
        if not self.spawn_package(pkg):
            return False
        time.sleep(2.5)

        pkg_idx = int(re.search(r"(\d+)$", pkg.id).group(1))

        # 2. Rotate to HOME2
        self.move_joints(HOME2_Q, "2-home2")
        time.sleep(2.5)

        # 3. Reach pickup
        target_z = PICKUP_TOP_TF_Z + pkg.height
        target_pos = [0.5, 0.0, target_z]
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"[3-reach-pickup] H={pkg.height:.4f}  target_z={target_z:.4f}")
        solved = self.analytical_ik(
            target_pos, TOOL_DOWN_QUAT_XYZW, HOME2_Q)
        if solved is None:
            return False
        self.move_joints(solved, "3-reach-pickup")
        time.sleep(1.0)

        # 4. Set relpose
        self.publish_relpose(pkg_idx, pkg.height)

        # 5. Weld ON
        self._t_weld_on = time.time()
        self.set_weld(True)

        # 6. Return HOME2
        self.move_joints(HOME2_Q, "6-return-home2")
        time.sleep(2.5)

        # 7. Rotate HOME
        if not self.move_joints(HOME_Q, "7-home"):
            self.get_logger().error("7-home failed; aborting cycle")
            return False
        time.sleep(2.5)

        # 8. Placement (timed)
        t_plan_start = time.time()
        target = self.request_placement(pkg)
        t_plan_end = time.time()
        plan_time = t_plan_end - t_plan_start

        if target is None:
            return False
        pos_internal = np.array([
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        ])
        pkg_center_tf = pos_internal + CONTAINER_INTERNAL_TO_TF
        tool_target_tf = np.array([
            pkg_center_tf[0],
            pkg_center_tf[1],
            pkg_center_tf[2] + pkg.height / 2.0,
        ])
        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"[8-reach-place] tool_target_tf=("
            f"{tool_target_tf[0]:+.3f}, "
            f"{tool_target_tf[1]:+.3f}, "
            f"{tool_target_tf[2]:+.3f})")

        t_ik_start = time.time()
        solved_place = self.analytical_ik(
            tool_target_tf, TOOL_DOWN_QUAT_XYZW, HOME_Q)
        t_ik_end = time.time()
        ik_time = t_ik_end - t_ik_start

        if solved_place is None:
            return False
        self.move_joints(solved_place, "8-reach-place")
        time.sleep(1.0)

        # 9. Weld OFF (timed)
        t_weld_off = time.time()
        self.set_weld(False)
        time.sleep(1.0)   # initial release

        # 9a. Retract straight up ~5 cm so the falling package cannot disturb
        #     the arm as it settles in the container.
        retract_target = tool_target_tf.copy()
        retract_target[2] += 0.05
        solved_retract = self.analytical_ik(
            retract_target, TOOL_DOWN_QUAT_XYZW, solved_place)
        if solved_retract is not None:
            self.move_joints(solved_retract, "9-retract")
            time.sleep(0.8)
        else:
            self.get_logger().warn(
                "9-retract: no IK solution; skipping retract")

        # 9b. Notify container_state so the next cycle sees an updated grid
        self.publish_placement(target, pkg)
        t_placed = time.time()
        settle_time = t_placed - t_weld_off

        # 9c. Let the package finish bouncing inside the container before we
        #     ask the arm to move home.
        time.sleep(2.0)

        # 10. Return HOME
        #     The package is already placed and container_state has been
        #     notified, so a failure here does not lose any data. Log a
        #     warning and let the next cycle's 0-recover-home pull the arm
        #     back if needed.
        if not self.move_joints(HOME_Q, "10-home"):
            self.get_logger().warn(
                "10-home failed; next cycle will recover")
        time.sleep(1.5)

        # ---- Metrics ----
        carry_time = t_weld_off - self._t_weld_on
        cycle_time = time.time() - t_cycle_start

        self.get_logger().info("=" * 60)
        self.get_logger().info(
            f"[metrics] {pkg.id}  "
            f"plan={plan_time:.3f}s  ik={ik_time:.3f}s  "
            f"carry={carry_time:.3f}s  settle={settle_time:.3f}s  "
            f"cycle={cycle_time:.3f}s")
        self.get_logger().info(
            f"[metrics] target_container=("
            f"{pos_internal[0]:.4f}, {pos_internal[1]:.4f}, "
            f"{pos_internal[2]:.4f})  "
            f"tool_target=("
            f"{tool_target_tf[0]:.4f}, {tool_target_tf[1]:.4f}, "
            f"{tool_target_tf[2]:.4f})")

        try:
            with open(self.metrics_path, "a") as f:
                f.write(
                    f"{cycle_num},{pkg.id},"
                    f"{pos_internal[0]:.4f},{pos_internal[1]:.4f},"
                    f"{pos_internal[2]:.4f},"
                    f"{tool_target_tf[0]:.4f},{tool_target_tf[1]:.4f},"
                    f"{tool_target_tf[2]:.4f},"
                    f"{plan_time:.3f},{ik_time:.3f},"
                    f"{carry_time:.3f},{settle_time:.3f},"
                    f"{cycle_time:.3f}\n"
                )
        except OSError as e:
            self.get_logger().warn(f"Could not write metrics CSV: {e}")

        self.get_logger().info(
            f"Cycle {cycle_num} complete ({pkg.id})")
        return True


def main():
    rclpy.init()
    node = PickPlaceDriver()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run():
        time.sleep(2.0)
        success_count = 0

        for i in range(1, node.num_packages + 1):
            ok = node.run_one_cycle(i)
            if not ok:
                node.get_logger().error(
                    f"Cycle {i} failed; aborting")
                break
            success_count += 1
            time.sleep(1.0)

        node.get_logger().info("=" * 60)
        node.get_logger().info(
            f"All done: {success_count}/{node.num_packages} cycles succeeded")

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
