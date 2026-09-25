#!/usr/bin/env python3
"""
Print tool0, flange, and wrist_3_link poses in base_link for one joint
configuration. Compare with ssik's fk() on the same joints.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK
from ssik.prebuilt import ur10e_ik


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class Diag(Node):
    def __init__(self):
        super().__init__("diagnose_frames")
        self.cli = self.create_client(GetPositionFK, "/compute_fk")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_fk...")

    def fk(self, joints, links):
        js = JointState()
        js.name = list(JOINT_NAMES)
        js.position = [float(x) for x in joints]
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = links
        req.robot_state.joint_state = js
        fut = self.cli.call_async(req)
        deadline = __import__("time").time() + 5.0
        while not fut.done():
            if __import__("time").time() > deadline:
                return None
            __import__("time").sleep(0.02)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            return None
        return res.pose_stamped


def main():
    rclpy.init()
    node = Diag()

    # Pick one test joint configuration — HOME2
    q = [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

    print()
    print("=" * 70)
    print(f"Joint configuration: {[f'{v:+.3f}' for v in q]}")
    print("=" * 70)

    # MuJoCo / URDF FK
    links = ["tool0", "flange", "wrist_3_link"]
    poses = node.fk(q, links)
    if poses is None:
        print("FK failed")
        return

    print()
    print("MuJoCo / URDF FK in base_link:")
    for ps in poses:
        p = ps.pose.position
        print(f"  {ps.header.frame_id if False else '':12s}"
              f"  {ps.pose.position.x:+.4f}  "
              f"{ps.pose.position.y:+.4f}  "
              f"{ps.pose.position.z:+.4f}")
    print()
    for i, name in enumerate(links):
        p = poses[i].pose.position
        print(f"  {name:16s}  ({p.x:+.4f}, {p.y:+.4f}, {p.z:+.4f})")

    # ssik FK
    T_ssik = ur10e_ik.fk(np.asarray(q))
    print()
    print("ssik FK (in ssik's DH base frame):")
    print(f"  tool0            "
          f"({T_ssik[0,3]:+.4f}, {T_ssik[1,3]:+.4f}, {T_ssik[2,3]:+.4f})")
    print()
    print("Reminder: ssik's base is the URDF base rotated 180 deg about Z.")
    print("To compare, apply the rotation:")
    print(f"  tool0 in URDF    "
          f"({-T_ssik[0,3]:+.4f}, {-T_ssik[1,3]:+.4f}, {T_ssik[2,3]:+.4f})")

    # Print the site offset for reference
    print()
    print("=" * 70)
    print("Interpretation:")
    print("  - if ssik tool0 matches MuJoCo tool0 above, the frame is fine")
    print("  - the gap between flange and wrist_3_link is the flange offset")
    print("  - the weld relpose uses wrist_3_link, so the package hangs")
    print("    relative to the wrist body, not the flange")
    print("=" * 70)


if __name__ == "__main__":
    main()
