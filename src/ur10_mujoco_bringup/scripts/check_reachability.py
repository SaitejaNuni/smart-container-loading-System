#!/usr/bin/env python3
"""
Bare reachability check. No seeds, no constraints. Just ask MoveIt:
can IK solve each target at all?

If a target fails here, it is unreachable in the current robot/workspace
configuration. If it passes, the problem is branch selection, not reach.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
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
    # Pickup targets
    # Pickup station top in MJCF = z 0.75. In new base frame = 0.75 - 0.45 = 0.30.
    ("pickup_approach",
     [0.500, 0.000, 0.550, 0.000, 0.707, 0.000, 0.707]),
    ("pickup_grasp",
     [0.500, 0.000, 0.400, 0.000, 0.707, 0.000, 0.707]),

    # Container floor targets (MJCF z=0 -> base z=-0.45)
    ("container_far_corner_floor",
     [-0.900, -0.900, -0.450, 0.000, 0.707, 0.000, 0.707]),
    ("container_far_corner_mid",
     [-0.900, -0.900, 0.450, 0.000, 0.707, 0.000, 0.707]),
    ("container_center_floor",
     [-0.400, 0.000, -0.450, 0.000, 0.707, 0.000, 0.707]),
    ("container_near_corner_floor",
     [0.100, 0.900, -0.450, 0.000, 0.707, 0.000, 0.707]),
]


class ReachabilityChecker(Node):
    def __init__(self):
        super().__init__("reachability_checker")
        self.cli = self.create_client(GetPositionIK, "/compute_ik")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")
        self.get_logger().info("Connected to /compute_ik")

    def try_ik(self, pose):
        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 2

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose.position.x = pose[0]
        pose_stamped.pose.position.y = pose[1]
        pose_stamped.pose.position.z = pose[2]
        pose_stamped.pose.orientation.x = pose[3]
        pose_stamped.pose.orientation.y = pose[4]
        pose_stamped.pose.orientation.z = pose[5]
        pose_stamped.pose.orientation.w = pose[6]
        req.ik_request.pose_stamped = pose_stamped

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            return None
        if future.result() is None:
            return None
        if future.result().error_code.val != 1:
            return None

        js = future.result().solution.joint_state
        name_to_pos = dict(zip(js.name, js.position))
        return [name_to_pos.get(j, 0.0) for j in JOINT_NAMES]


def main():
    rclpy.init()
    node = ReachabilityChecker()

    print()
    print("=" * 70)
    print("Reachability Check (10 tries per target, no seed, no constraints)")
    print("=" * 70)

    for label, pose in TARGETS:
        print(f"\n{label}")
        print(f"  pose: x={pose[0]:+.3f}  y={pose[1]:+.3f}  z={pose[2]:+.3f}")
        dist = (pose[0] ** 2 + pose[1] ** 2 + pose[2] ** 2) ** 0.5
        print(f"  distance from base: {dist:.3f} m")

        successes = 0
        unique = set()
        for i in range(10):
            joints = node.try_ik(pose)
            if joints is not None:
                successes += 1
                unique.add(tuple(round(j, 2) for j in joints))

        print(f"  IK attempts: 10")
        print(f"  Successful: {successes}")
        print(f"  Distinct solutions: {len(unique)}")
        if successes > 0:
            print(f"  Verdict: REACHABLE")
        else:
            print(f"  Verdict: UNREACHABLE in current configuration")

    print()
    print("=" * 70)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
