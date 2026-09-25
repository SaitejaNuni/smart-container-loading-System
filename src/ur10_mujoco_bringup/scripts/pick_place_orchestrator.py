#!/usr/bin/env python3
"""
Pick-and-place orchestrator (clean version).

Strategy — the "two-home-pose" method:

    HOME   : arm inverted-L, facing the CONTAINER   (shoulder_pan = 0)
    HOME2  : arm inverted-L, facing the PICKUP      (shoulder_pan = pi)

Sequence per package:

    START (whatever pose the arm is in)
        |
        v  joint-space (move_to_configuration)
    HOME
        |
        v  joint-space 180 deg rotation
    HOME2
        |
        v  pose move (IK + RRTstar)
    Reach over the pickup station
        |
        v  pose move (IK + RRTstar)
    Descend to the package
        |
        v  weld ON
    Retract to HOME2
        |
        v  joint-space 180 deg rotation
    HOME
        |
        v  pose move (IK + RRTstar)
    Reach over placement target
        |
        v  pose move (IK + RRTstar)
    Descend and release
        |
        v  weld OFF
    Return to HOME

Only the three "reach" and "descend" steps use IK + RRTstar.
Every other step is a direct joint-space move with no IK.
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
from thesis_msgs.srv import GetNextPackage, PlanPlacement, PlaceBox

from pymoveit2 import MoveIt2


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# =============================================================================
# Frame constants
# =============================================================================

# The MJCF world places the robot base at (0.9, 0, 0.45).
# TF world places base_link at origin.
# So: TF_pos = MJCF_pos - (0.9, 0, 0.45)
MJCF_WORLD_TO_TF = np.array([0.9, 0.0, 0.45])

# Container-internal coordinates use origin at back-left-bottom corner.
# MJCF world = internal + (0, -0.9, 0)
CONTAINER_ORIGIN_MJCF = np.array([0.0, -0.9, 0.0])

# =============================================================================
# Home poses (joint space)
# =============================================================================
#
# Inverted L: upper arm horizontal, forearm bent 90 degrees up.
# The only difference between HOME and HOME2 is shoulder_pan.
#
# If the arm faces the wrong way, swap the first value of each array.

HOME_FACING_CONTAINER = np.array([
    0.0,        # shoulder_pan
    -1.5708,    # shoulder_lift  (upper arm horizontal)
    1.5708,     # elbow          (forearm bent up)
    -1.5708,    # wrist_1
    -1.5708,    # wrist_2
    0.0,        # wrist_3
])

HOME_FACING_PICKUP = np.array([
    3.1416,     # shoulder_pan rotated 180 deg
    -1.5708,
    1.5708,
    -1.5708,
    -1.5708,
    0.0,
])

# =============================================================================
# Pose constants
# =============================================================================

# Tool pointing straight down (ROS xyzw quaternion).
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# Pickup station top surface in MJCF world.
PICKUP_TOP_MJCF = np.array([1.4, 0.0, 0.75])

# Weld geometry. Package hangs 0.23 m from the wrist along the wrist local
# +X axis (matches the MJCF weld relpose).
WELD_OFFSET_FROM_WRIST = 0.23


# =============================================================================
# State machine
# =============================================================================

class State:
    IDLE = "IDLE"
    GET_PACKAGE = "GET_PACKAGE"
    SPAWN = "SPAWN"
    GO_HOME = "GO_HOME"
    ROTATE_TO_PICKUP = "ROTATE_TO_PICKUP"
    APPROACH_PICKUP = "APPROACH_PICKUP"
    GRASP = "GRASP"
    RETURN_TO_HOME2 = "RETURN_TO_HOME2"
    ROTATE_TO_CONTAINER = "ROTATE_TO_CONTAINER"
    PLAN_PLACEMENT = "PLAN_PLACEMENT"
    APPROACH_PLACEMENT = "APPROACH_PLACEMENT"
    RELEASE = "RELEASE"
    RETURN_TO_HOME = "RETURN_TO_HOME"
    CONFIRM = "CONFIRM"
    DONE = "DONE"


# =============================================================================
# Orchestrator
# =============================================================================

class Orchestrator(Node):

    def __init__(self):
        super().__init__("pick_place_orchestrator")

        # ---------------- Parameters ----------------
        self.declare_parameter("num_packages", 1)
        self.declare_parameter("settle_after_grasp_sec", 1.0)
        self.declare_parameter("settle_after_release_sec", 1.5)
        self.declare_parameter("state_dwell_sec", 0.5)
        self.declare_parameter("pre_grasp_height", 0.15)
        self.declare_parameter("pre_place_height", 0.15)
        self.declare_parameter("disable_weld", False)

        self.declare_parameter("max_velocity", 0.5)
        self.declare_parameter("max_acceleration", 0.5)

        self.declare_parameter("srv_get_next_package", "/get_next_package")
        self.declare_parameter("srv_plan_placement", "/plan_placement")
        self.declare_parameter("srv_place_box", "/place_box")
        self.declare_parameter("srv_spawn_package", "/spawn_package")

        # ---------------- Read parameters ----------------
        self.num_packages = self.get_parameter("num_packages").value
        self.settle_after_grasp_sec = self.get_parameter(
            "settle_after_grasp_sec").value
        self.settle_after_release_sec = self.get_parameter(
            "settle_after_release_sec").value
        self.state_dwell_sec = self.get_parameter("state_dwell_sec").value
        self.pre_grasp_height = self.get_parameter("pre_grasp_height").value
        self.pre_place_height = self.get_parameter("pre_place_height").value
        self.disable_weld = self.get_parameter("disable_weld").value

        # ---------------- ROS interface ----------------
        self.cb = ReentrantCallbackGroup()

        self.active_idx_pub = self.create_publisher(
            Int32, "/active_package_index", 10)
        self.gripper_pub = self.create_publisher(
            Bool, "/gripper_command", 10)

        self.cli_get_next = self.create_client(
            GetNextPackage,
            self.get_parameter("srv_get_next_package").value,
            callback_group=self.cb)
        self.cli_plan = self.create_client(
            PlanPlacement,
            self.get_parameter("srv_plan_placement").value,
            callback_group=self.cb)
        self.cli_place = self.create_client(
            PlaceBox,
            self.get_parameter("srv_place_box").value,
            callback_group=self.cb)
        self.cli_spawn = self.create_client(
            Trigger,
            self.get_parameter("srv_spawn_package").value,
            callback_group=self.cb)

        # ---------------- MoveIt2 ----------------
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=JOINT_NAMES,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            callback_group=self.cb,
        )
        self.moveit2.max_velocity = self.get_parameter("max_velocity").value
        self.moveit2.max_acceleration = self.get_parameter(
            "max_acceleration").value
        self.moveit2.planner_id = "ur_manipulator[RRTstar]"

        # ---------------- State ----------------
        self.state = State.IDLE
        self.packages_done = 0
        self.current_pkg = None
        self.current_pkg_index = None
        self.current_placement = None

        self.get_logger().info(
            "Orchestrator constructed. "
            f"planner = ur_manipulator[RRTstar], "
            f"MJCF->TF offset = {MJCF_WORLD_TO_TF.tolist()}"
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    def set_state(self, s):
        self.get_logger().info(f"[STATE] {self.state} -> {s}")
        self.state = s

    def wait_for_services(self, timeout=30.0):
        pairs = [
            (self.cli_get_next, "GetNextPackage"),
            (self.cli_plan, "PlanPlacement"),
            (self.cli_place, "PlaceBox"),
            (self.cli_spawn, "SpawnPackage"),
        ]
        for cli, name in pairs:
            if not cli.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f"Service {name} not available")
                return False
        return True

    def call_service(self, client, request, timeout=15.0):
        fut = client.call_async(request)
        deadline = time.time() + timeout
        while not fut.done():
            if time.time() > deadline:
                self.get_logger().warn("Service call timed out")
                return None
            time.sleep(0.02)
        return fut.result()

    def move_to_configuration(self, joints, retries=3):
        """Direct joint-space move. No IK, no path planning shortcuts."""
        self.get_logger().info(
            f"Move joints: {[f'{j:+.3f}' for j in joints]}")
        for attempt in range(retries + 1):
            self.moveit2.move_to_configuration(list(joints))
            if self.moveit2.wait_until_executed():
                return True
            self.get_logger().warn(
                f"Joint move attempt {attempt + 1} failed")
            time.sleep(0.5)
        return False

    def move_to_pose(self, pos_tf, quat_xyzw, retries=3):
        """IK + RRTstar pose move."""
        pos_list = [float(x) for x in pos_tf]
        quat_list = [float(x) for x in quat_xyzw]
        self.get_logger().info(
            f"Move pose: ({pos_list[0]:+.3f}, "
            f"{pos_list[1]:+.3f}, {pos_list[2]:+.3f})")
        for attempt in range(retries + 1):
            self.moveit2.move_to_pose(
                position=pos_list,
                quat_xyzw=quat_list,
            )
            if self.moveit2.wait_until_executed():
                return True
            self.get_logger().warn(
                f"Pose move attempt {attempt + 1} failed")
            time.sleep(0.5)
        return False

    def mjcf_to_tf(self, p):
        return np.array(p) - MJCF_WORLD_TO_TF

    def container_internal_to_mjcf(self, p):
        return np.array(p) + CONTAINER_ORIGIN_MJCF

    def make_package_index(self, package_id):
        try:
            return int(package_id.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    # -------------------------------------------------------------------------
    # Target pose computation
    # -------------------------------------------------------------------------

    def pickup_approach_pose(self):
        """TCP pose in TF world, above the package at the pickup station."""
        h = self.current_pkg.height
        pkg_center_mjcf = PICKUP_TOP_MJCF + np.array([0.0, 0.0, h / 2.0])
        tcp_mjcf = pkg_center_mjcf + np.array(
            [0.0, 0.0, WELD_OFFSET_FROM_WRIST + self.pre_grasp_height])
        return self.mjcf_to_tf(tcp_mjcf)

    def pickup_grasp_pose(self):
        """TCP pose in TF world, at the grasp height."""
        h = self.current_pkg.height
        pkg_center_mjcf = PICKUP_TOP_MJCF + np.array([0.0, 0.0, h / 2.0])
        tcp_mjcf = pkg_center_mjcf + np.array(
            [0.0, 0.0, WELD_OFFSET_FROM_WRIST])
        return self.mjcf_to_tf(tcp_mjcf)

    def placement_approach_pose(self):
        """TCP pose and package center, both in TF and MJCF respectively."""
        pose = self.current_placement.pose
        pos_internal = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        pkg_center_mjcf = self.container_internal_to_mjcf(pos_internal)
        tcp_mjcf = pkg_center_mjcf + np.array(
            [0.0, 0.0, WELD_OFFSET_FROM_WRIST + self.pre_place_height])
        return self.mjcf_to_tf(tcp_mjcf), pkg_center_mjcf

    def placement_release_pose(self):
        pose = self.current_placement.pose
        pos_internal = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        pkg_center_mjcf = self.container_internal_to_mjcf(pos_internal)
        tcp_mjcf = pkg_center_mjcf + np.array(
            [0.0, 0.0, WELD_OFFSET_FROM_WRIST])
        return self.mjcf_to_tf(tcp_mjcf), pkg_center_mjcf

    # =========================================================================
    # State handlers
    # =========================================================================

    def handle_get_package(self):
        res = self.call_service(
            self.cli_get_next, GetNextPackage.Request(), timeout=5.0)
        if res is None or not res.has_next:
            self.get_logger().info("No more packages.")
            self.set_state(State.DONE)
            return
        self.current_pkg = res.package
        self.current_pkg_index = self.make_package_index(
            self.current_pkg.id)
        if self.current_pkg_index is None:
            self.get_logger().error(
                f"Cannot parse package id '{self.current_pkg.id}'")
            self.set_state(State.DONE)
            return
        self.get_logger().info(
            f"Got {self.current_pkg.id}: "
            f"L={self.current_pkg.length:.3f} "
            f"W={self.current_pkg.width:.3f} "
            f"H={self.current_pkg.height:.3f}")
        self.set_state(State.SPAWN)

    def handle_spawn(self):
        msg = Int32()
        msg.data = self.current_pkg_index
        self.active_idx_pub.publish(msg)
        time.sleep(0.15)

        res = self.call_service(
            self.cli_spawn, Trigger.Request(), timeout=5.0)
        if res is None or not res.success:
            self.get_logger().error("spawn_package failed")
            self.set_state(State.DONE)
            return
        self.set_state(State.GO_HOME)

    def handle_go_home(self):
        """Joint-space move to HOME facing the container."""
        if not self.move_to_configuration(HOME_FACING_CONTAINER):
            self.get_logger().error("Failed to reach HOME")
            self.set_state(State.DONE)
            return
        self.set_state(State.ROTATE_TO_PICKUP)

    def handle_rotate_to_pickup(self):
        """Joint-space 180 degree shoulder rotation. No IK."""
        if not self.move_to_configuration(HOME_FACING_PICKUP):
            self.get_logger().error("Failed to reach HOME2")
            self.set_state(State.DONE)
            return
        self.set_state(State.APPROACH_PICKUP)

    def handle_approach_pickup(self):
        if not self.move_to_pose(
                self.pickup_approach_pose(), TOP_DOWN_QUAT):
            self.get_logger().error("Approach pickup failed")
            self.set_state(State.DONE)
            return
        self.set_state(State.GRASP)

    def handle_grasp(self):
        if not self.move_to_pose(
                self.pickup_grasp_pose(), TOP_DOWN_QUAT):
            self.get_logger().error("Descend to grasp failed")
            self.set_state(State.DONE)
            return

        if self.disable_weld:
            self.get_logger().info("Gripper CLOSED (disabled)")
        else:
            g = Bool()
            g.data = True
            self.gripper_pub.publish(g)
            self.get_logger().info("Gripper CLOSED (weld ON)")
        time.sleep(self.settle_after_grasp_sec)
        self.set_state(State.RETURN_TO_HOME2)

    def handle_return_to_home2(self):
        """Joint-space return to HOME2 while holding the package."""
        if not self.move_to_configuration(HOME_FACING_PICKUP):
            self.get_logger().warn("Return to HOME2 failed; continuing")
        self.set_state(State.ROTATE_TO_CONTAINER)

    def handle_rotate_to_container(self):
        """Joint-space 180 degree rotation back to HOME."""
        if not self.move_to_configuration(HOME_FACING_CONTAINER):
            self.get_logger().warn("Rotate to HOME failed; continuing")
        self.set_state(State.PLAN_PLACEMENT)

    def handle_plan_placement(self):
        req = PlanPlacement.Request()
        req.package = self.current_pkg
        res = self.call_service(self.cli_plan, req, timeout=15.0)
        if res is None or not res.success:
            reason = res.reason if res else "no response"
            self.get_logger().error(f"Placement planning failed: {reason}")
            self.set_state(State.DONE)
            return
        self.current_placement = res.target
        self.get_logger().info(
            f"Placement planned (container-internal): "
            f"({res.target.pose.position.x:.3f}, "
            f"{res.target.pose.position.y:.3f}, "
            f"{res.target.pose.position.z:.3f})")
        self.set_state(State.APPROACH_PLACEMENT)

    def handle_approach_placement(self):
        tcp_tf, pkg_center_mjcf = self.placement_approach_pose()
        self.get_logger().info(
            f"Package center (MJCF world): "
            f"({pkg_center_mjcf[0]:+.3f}, "
            f"{pkg_center_mjcf[1]:+.3f}, "
            f"{pkg_center_mjcf[2]:+.3f})")
        if not self.move_to_pose(tcp_tf, TOP_DOWN_QUAT):
            self.get_logger().error("Approach placement failed")
            self.set_state(State.DONE)
            return
        self.set_state(State.RELEASE)

    def handle_release(self):
        tcp_tf, pkg_center_mjcf = self.placement_release_pose()
        if not self.move_to_pose(tcp_tf, TOP_DOWN_QUAT):
            self.get_logger().error("Descend to place failed")
            self.set_state(State.DONE)
            return

        if self.disable_weld:
            self.get_logger().info("Gripper OPENED (disabled)")
        else:
            g = Bool()
            g.data = False
            self.gripper_pub.publish(g)
            self.get_logger().info("Gripper OPENED (weld OFF)")

        time.sleep(self.settle_after_release_sec)

        # Notify container state node
        pos_internal = pkg_center_mjcf - CONTAINER_ORIGIN_MJCF
        req = PlaceBox.Request()
        req.package_id = self.current_pkg.id
        req.x = float(pos_internal[0])
        req.y = float(pos_internal[1])
        req.z = float(pos_internal[2])
        req.length = self.current_pkg.length
        req.width = self.current_pkg.width
        req.height = self.current_pkg.height
        req.mass = self.current_pkg.mass
        res = self.call_service(self.cli_place, req, timeout=5.0)
        if res is None or not res.success:
            msg = res.error_message if res else "no response"
            self.get_logger().warn(f"place_box failure: {msg}")
        else:
            self.get_logger().info("Container state updated")

        self.set_state(State.RETURN_TO_HOME)

    def handle_return_to_home(self):
        if not self.move_to_configuration(HOME_FACING_CONTAINER):
            self.get_logger().warn("Return to HOME failed")
        self.set_state(State.CONFIRM)

    def handle_confirm(self):
        self.packages_done += 1
        self.get_logger().info(
            f"Cycle {self.packages_done}/{self.num_packages} "
            f"complete ({self.current_pkg.id})")
        time.sleep(self.state_dwell_sec)
        if self.packages_done >= self.num_packages:
            self.set_state(State.DONE)
        else:
            self.set_state(State.GET_PACKAGE)

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self):
        if not self.wait_for_services(timeout=30.0):
            self.get_logger().error("Required services unavailable")
            return

        self.get_logger().info(
            f"Starting orchestration for {self.num_packages} package(s)")
        self.set_state(State.GET_PACKAGE)

        handlers = {
            State.GET_PACKAGE: self.handle_get_package,
            State.SPAWN: self.handle_spawn,
            State.GO_HOME: self.handle_go_home,
            State.ROTATE_TO_PICKUP: self.handle_rotate_to_pickup,
            State.APPROACH_PICKUP: self.handle_approach_pickup,
            State.GRASP: self.handle_grasp,
            State.RETURN_TO_HOME2: self.handle_return_to_home2,
            State.ROTATE_TO_CONTAINER: self.handle_rotate_to_container,
            State.PLAN_PLACEMENT: self.handle_plan_placement,
            State.APPROACH_PLACEMENT: self.handle_approach_placement,
            State.RELEASE: self.handle_release,
            State.RETURN_TO_HOME: self.handle_return_to_home,
            State.CONFIRM: self.handle_confirm,
        }

        while rclpy.ok() and self.state != State.DONE:
            handler = handlers.get(self.state)
            if handler is None:
                self.get_logger().error(
                    f"No handler for state {self.state}")
                break
            try:
                handler()
            except Exception as e:
                self.get_logger().error(f"Handler failed: {e}")
                self.set_state(State.DONE)

        self.get_logger().info("Orchestrator finished.")


def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # Run the state machine in a background thread, spin the executor in main.
    t = threading.Thread(target=node.run, daemon=True)
    t.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Avoid the "rcl_shutdown already called" error by checking state first.
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
