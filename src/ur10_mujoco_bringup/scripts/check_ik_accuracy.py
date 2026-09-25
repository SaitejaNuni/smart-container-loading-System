#!/usr/bin/env python3
"""
IK accuracy check: for each target, solve IK, then run FK on the
returned joints and compare with the requested pose.

This closes the loop: it verifies that pick_ik returns joint angles that
actually place the end-effector at the requested pose, within tolerance.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import PositionIKRequest


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
    ("container_far_corner_floor",
     [-0.900, -0.900, -0.450, 0.000, 0.707, 0.000, 0.707]),
]


def quat_to_rot(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def orientation_error_deg(q1, q2):
    """Angle between two quaternions in degrees."""
    dot = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class IKAccuracyChecker(Node):
    def __init__(self):
        super().__init__("ik_accuracy_checker")

        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        self.fk_cli = self.create_client(GetPositionFK, "/compute_fk")

        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")
        while not self.fk_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_fk...")
        self.get_logger().info("Connected to /compute_ik and /compute_fk")

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
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            return None
        if future.result().error_code.val != 1:
            return None
        return future.result().solution.joint_state

    def compute_fk(self, joint_state):
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state.joint_state = joint_state

        future = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            return None
        if future.result().error_code.val != 1:
            return None
        if len(future.result().pose_stamped) == 0:
            return None
        return future.result().pose_stamped[0].pose


def main():
    rclpy.init()
    node = IKAccuracyChecker()

    print()
    print("=" * 78)
    print("IK Accuracy Check: requested pose vs FK of IK solution")
    print("=" * 78)

    for label, target in TARGETS:
        print(f"\n--- {label} ---")
        print(f"  Requested position:  x={target[0]:+.4f}  y={target[1]:+.4f}  z={target[2]:+.4f}")
        print(f"  Requested orient. :  qx={target[3]:+.4f} qy={target[4]:+.4f} qz={target[5]:+.4f} qw={target[6]:+.4f}")

        js = node.solve_ik(target)
        if js is None:
            print("  IK: FAILED (no solution)")
            continue

        joint_vals = dict(zip(js.name, js.position))
        joints_str = "[" + ", ".join(
            f"{joint_vals.get(j, 0.0):+.3f}" for j in JOINT_NAMES
        ) + "]"
        print(f"  IK solution:         {joints_str}")

        achieved = node.compute_fk(js)
        if achieved is None:
            print("  FK: FAILED")
            continue

        p = achieved.position
        q = achieved.orientation
        print(f"  Achieved position:   x={p.x:+.4f}  y={p.y:+.4f}  z={p.z:+.4f}")
        print(f"  Achieved orient. :   qx={q.x:+.4f} qy={q.y:+.4f} qz={q.z:+.4f} qw={q.w:+.4f}")

        dpos = math.sqrt(
            (p.x - target[0]) ** 2
            + (p.y - target[1]) ** 2
            + (p.z - target[2]) ** 2
        )
        dang = orientation_error_deg(
            (q.x, q.y, q.z, q.w),
            (target[3], target[4], target[5], target[6]),
        )

        print(f"  POSITION ERROR:      {dpos * 1000:.3f} mm")
        print(f"  ORIENTATION ERROR:   {dang:.3f} deg")

        if dpos < 0.005 and dang < 3.0:
            print("  VERDICT: ACCURATE (within tolerance)")
        elif dpos < 0.020 and dang < 10.0:
            print("  VERDICT: ACCEPTABLE (a bit off, but usable)")
        else:
            print("  VERDICT: POOR (outside reasonable tolerance)")

    print()
    print("=" * 78)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
