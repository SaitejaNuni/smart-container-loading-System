#!/usr/bin/env python3
"""
Publish the MJCF container to MoveIt's planning scene.

Two parameters control the conversion:
  offset_xyz (float array, default [0.9, 0.0, 0.45]) — MJCF -> TF offset
  flip_y     (bool, default False)                  — flip the Y axis

Run:  ros2 run ur10_mujoco_bringup publish_container_to_moveit.py
      ros2 run ur10_mujoco_bringup publish_container_to_moveit.py \
          --ros-args -p flip_y:=true
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive


class ContainerPublisher(Node):
    def __init__(self):
        super().__init__("publish_container_to_moveit")

        self.declare_parameter("offset_xyz", [0.9, 0.0, 0.45])
        self.declare_parameter("flip_y", False)

        self.offset = self.get_parameter("offset_xyz").value
        self.flip_y = self.get_parameter("flip_y").value

        self.pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.timer = self.create_timer(1.0, self.publish_once)
        self.count = 0
        self.max_publishes = 3

        self.get_logger().info(
            f"Container publisher ready. offset={self.offset}, "
            f"flip_y={self.flip_y}")

    def transform(self, mjcf_xyz):
        x = mjcf_xyz[0] - self.offset[0]
        y = mjcf_xyz[1] - self.offset[1]
        z = mjcf_xyz[2] - self.offset[2]
        if self.flip_y:
            y = -y
        return (x, y, z)

    def make_box(self, name, mjcf_xyz, size):
        x, y, z = self.transform(mjcf_xyz)
        obj = CollisionObject()
        obj.header.frame_id = "world"
        obj.id = name
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [size[0], size[1], size[2]]
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        self.get_logger().info(
            f"  {name}: MJCF {mjcf_xyz} -> TF ({x:+.3f}, {y:+.3f}, {z:+.3f})")
        return obj

    def publish_once(self):
        scene = PlanningScene()
        scene.is_diff = True

        # Container interior: x in [0, 1.0], y in [-0.9, 0.9], z in [0, 1.8]
        # Wall thickness: 0.01
        scene.world.collision_objects.append(
            self.make_box("container_back", [0.0, 0.0, 0.9],
                          [0.01, 1.8, 1.8]))
        scene.world.collision_objects.append(
            self.make_box("container_left", [0.5, -0.9, 0.9],
                          [1.0, 0.01, 1.8]))
        scene.world.collision_objects.append(
            self.make_box("container_right", [0.5, 0.9, 0.9],
                          [1.0, 0.01, 1.8]))
        scene.world.collision_objects.append(
            self.make_box("container_top", [0.5, 0.0, 1.8],
                          [1.0, 1.8, 0.01]))
        scene.world.collision_objects.append(
            self.make_box("container_floor", [0.5, 0.0, 0.0],
                          [1.0, 1.8, 0.01]))

        self.pub.publish(scene)
        self.count += 1
        self.get_logger().info(
            f"Published {len(scene.world.collision_objects)} geoms "
            f"(attempt {self.count})")
        if self.count >= self.max_publishes:
            self.timer.cancel()


def main():
    rclpy.init()
    node = ContainerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
