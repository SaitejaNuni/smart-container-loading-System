#!/usr/bin/env python3
"""
Inspect the GetState response from container_state_node and print every
field so we can determine the container-internal frame direction.
"""

import rclpy
from rclpy.node import Node
from thesis_msgs.srv import GetState


class ContainerFrameChecker(Node):
    def __init__(self):
        super().__init__("container_frame_checker")
        self.cli = self.create_client(GetState, "/get_state")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /get_state...")
        self.get_logger().info("Connected to /get_state")

    def query(self):
        req = GetState.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return None
        return future.result()


def dump_fields(obj, prefix=""):
    """Print every attribute and its value, or length if iterable."""
    if not hasattr(obj, "__slots__") and not hasattr(obj, "__dict__"):
        print(f"{prefix}{obj}")
        return
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        if isinstance(val, (int, float, str, bool)):
            print(f"{prefix}{name} = {val}")
        elif hasattr(val, "__len__"):
            print(f"{prefix}{name}: len={len(val)}")
            for i, item in enumerate(list(val)[:8]):
                print(f"{prefix}  [{i}] {item}")
        else:
            print(f"{prefix}{name} = <object of type {type(val).__name__}>")


def main():
    rclpy.init()
    node = ContainerFrameChecker()

    res = node.query()
    if res is None:
        print("GetState failed")
        rclpy.shutdown()
        return

    print()
    print("=" * 60)
    print("GetState response contents")
    print("=" * 60)
    dump_fields(res)

    print()
    print("=" * 60)
    print("Container spans internally (from bounds check in the node):")
    print("  x in [0, 1.0]   -> MJCF x in [0, 1.0]   (shift  0.0)")
    print("  y in [0, 1.8]   -> MJCF y in [-0.9, 0.9]")
    print("  z in [0, 1.8]   -> MJCF z in [0, 1.8]")
    print()
    print("Look at the 'poses' field values (y coordinate):")
    print("  If all internal y are positive small values (0 to 1.8):")
    print("      container_internal + (0, -0.9, 0) = MJCF world")
    print("  If any y is negative:")
    print("      the node may already be using MJCF world frame")
    print("=" * 60)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
