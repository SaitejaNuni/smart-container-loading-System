#!/usr/bin/env python3
"""
For each candidate orientation quaternion, solve IK for the pickup
position and compute the resulting tool Z axis direction in base_link.

The tool should point "down" if its Z axis points along (0, 0, -1).
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


POSITION = [0.500, 0.000, 0.550]

CANDIDATES = [
    ("identity",        [0.0,   0.0,   0.0,   1.0]),
    ("180_about_X",     [1.0,   0.0,   0.0,   0.0]),
    ("180_about_Y",     [0.0,   1.0,   0.0,   0.0]),
    ("180_about_Z",     [0.0,   0.0,   1.0,   0.0]),
    ("90_about_X",      [0.707, 0.707, 0.0,   0.0]),
    ("90_about_Y",      [0.0,   0.707, 0.0,   0.707]),
    ("90_about_Z",      [0.0,   0.0,   0.707, 0.707]),
    ("120_111",         [0.5,   0.5,   0.5,   0.5]),
]


def quat_to_rot(qx, qy, qz, qw):
    return [
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ]


class Checker(Node):
    def __init__(self):
        super().__init__("check_tool_orientation")
        self.cb = ReentrantCallbackGroup()
        self.ik = self.create_client(GetPositionIK, "/compute_ik", callback_group=self.cb)
        self.fk = self.create_client(GetPositionFK, "/compute_fk", callback_group=self.cb)
        while not self.ik.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for /compute_ik...")
        while not self.fk.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for /compute_fk...")

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
        fut = self.ik.call_async(req)
        deadline = time.time() + 4.0
        while not fut.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        if fut.result() is None or fut.result().error_code.val != 1:
            return None
        js = fut.result().solution.joint_state
        return js

    def compute_fk(self, js):
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state.joint_state = js
        fut = self.fk.call_async(req)
        deadline = time.time() + 4.0
        while not fut.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        if fut.result() is None:
            return None
        if len(fut.result().pose_stamped) == 0:
            return None
        return fut.result().pose_stamped[0].pose


def main():
    rclpy.init()
    node = Checker()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def run():
        print()
        print("=" * 78)
        print("Candidate quaternion test at position (0.5, 0, 0.55)")
        print("Tool Z axis in base_link  —  (0,0,-1) means pointing DOWN")
        print("=" * 78)

        for name, quat in CANDIDATES:
            pose = list(POSITION) + list(quat)
            js = node.solve_ik(pose)
            if js is None:
                print(f"\n{name:15s}  quat={quat}   IK FAILED")
                continue
            achieved = node.compute_fk(js)
            if achieved is None:
                print(f"\n{name:15s}  quat={quat}   FK FAILED")
                continue

            q = achieved.orientation
            R = quat_to_rot(q.x, q.y, q.z, q.w)
            tool_z = (R[0][2], R[1][2], R[2][2])
            tool_x = (R[0][0], R[1][0], R[2][0])

            downward = tool_z[2] < -0.9
            marker = "*** TOP-DOWN ***" if downward else ""

            print(f"\n{name:15s}  quat={quat}   {marker}")
            print(f"    tool Z axis: ({tool_z[0]:+.3f}, {tool_z[1]:+.3f}, {tool_z[2]:+.3f})")
            print(f"    tool X axis: ({tool_x[0]:+.3f}, {tool_x[1]:+.3f}, {tool_x[2]:+.3f})")

        print()
        print("Look for the candidate whose tool Z axis is closest to (0,0,-1).")
        print("That is the correct top-down quaternion for the demo.")
        print("=" * 78)

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
