"""
Visualization Node.

Publishes RViz markers for the container walls (static) and each placed
package (dynamic).

Container dimensions are ROS parameters, defaulting to the same values used
by container_state_node and placement_planner_node (1.0 x 1.8 x 1.8 m).
On startup the node queries container_state_node's actual parameters and
logs an ERROR if they disagree — a silent mismatch previously caused RViz
to draw a container of the wrong size.

Uses MultiThreadedExecutor + ReentrantCallbackGroup because service calls
happen from within callbacks (and during startup).

On startup:
  - Container walls published
  - Container dimensions verified against container_state_node
  - GetState called to fetch existing boxes
  - All markers published

On new placement:
  - Subscription callback fires
  - GetState called to fetch full state (for dimensions)
  - New marker added, all markers republished

Parameters:
  container_length (float, default 1.0): container X dimension (m)
  container_width  (float, default 1.8): container Y dimension (m)
  container_height (float, default 1.8): container Z dimension (m)
  wall_thickness   (float, default 0.02): visual wall thickness (m)
  verify_container_dims (bool, default True): cross-check against
    container_state_node's parameters on startup
"""

import hashlib
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.srv import GetParameters
from thesis_msgs.msg import PlacementTarget
from thesis_msgs.srv import GetState


FRAME_ID = 'world'
NS_CONTAINER = 'container'
NS_PACKAGES = 'packages'


class VisualizationNode(Node):
    """Publishes RViz markers for container and placed packages."""

    def __init__(self):
        super().__init__('visualization_node')

        # ---- Parameters (must match container_state_node) ----
        self.declare_parameter('container_length', 1.0)
        self.declare_parameter('container_width', 1.8)
        self.declare_parameter('container_height', 1.8)
        self.declare_parameter('wall_thickness', 0.02)
        self.declare_parameter('verify_container_dims', True)

        self.L = float(self.get_parameter('container_length').value)
        self.W = float(self.get_parameter('container_width').value)
        self.H = float(self.get_parameter('container_height').value)
        self.T = float(self.get_parameter('wall_thickness').value)
        self.verify_dims = self.get_parameter('verify_container_dims').value

        # Reentrant callback group so service calls from subscription work
        self.cb_group = ReentrantCallbackGroup()

        # State
        self.box_markers = {}
        self.container_markers = self._build_container_markers()

        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray, '/visualization_markers', 10
        )

        # Service client
        self.get_state_client = self.create_client(
            GetState, '/get_state',
            callback_group=self.cb_group
        )

        # Subscription
        self.placement_sub = self.create_subscription(
            PlacementTarget, '/placement_target',
            self.on_placement_received, 10,
            callback_group=self.cb_group
        )

        self.get_logger().info(
            f'Visualization node initialized. '
            f'Container: {self.L}m x {self.W}m x {self.H}m'
        )

        # Kick off initial state fetch in a background thread so that
        # __init__ can return and the executor can start spinning.
        self.startup_thread = threading.Thread(
            target=self._startup_sequence, daemon=True
        )
        self.startup_thread.start()

    # ==================== Startup ====================

    def _startup_sequence(self):
        """Wait for service, verify dims, fetch initial state, publish."""
        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /get_state...')

        if self.verify_dims:
            self._verify_container_dims()

        self.get_logger().info('Fetching initial state...')
        response = self.get_state_client.call(GetState.Request())
        for i, pkg in enumerate(response.packages):
            pose = response.poses[i]
            self._add_box_marker(
                package_id=pkg.id,
                corner_x=pose.position.x,
                corner_y=pose.position.y,
                corner_z=pose.position.z,
                length=pkg.length,
                width=pkg.width,
                height=pkg.height,
            )

        self._publish_all()
        self.get_logger().info(
            f'Visualization ready. Container: '
            f'{self.L}m x {self.W}m x {self.H}m. '
            f'Existing boxes: {len(self.box_markers)}'
        )

    def _verify_container_dims(self):
        """Cross-check our container dims against container_state_node's.

        A silent mismatch here draws a container of the wrong size in RViz
        while everything else appears to work, so fail loudly instead.
        """
        try:
            cli = self.create_client(
                GetParameters, '/container_state/get_parameters',
                callback_group=self.cb_group
            )
            if not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(
                    'Could not reach /container_state/get_parameters; '
                    'container dimensions not verified'
                )
                return

            req = GetParameters.Request()
            req.names = ['container_length', 'container_width',
                         'container_height']
            res = cli.call(req)
            theirs = [v.double_value for v in res.values]
            mine = [self.L, self.W, self.H]

            if any(abs(a - b) > 1e-6 for a, b in zip(mine, theirs)):
                self.get_logger().error(
                    f'CONTAINER SIZE MISMATCH — RViz will be wrong. '
                    f'visualization_node={mine} container_state={theirs}. '
                    f'Set both from the same launch file.'
                )
            else:
                self.get_logger().info(
                    f'Container dimensions verified against container_state: '
                    f'{theirs}'
                )
        except Exception as e:
            self.get_logger().warn(f'Could not verify container dims: {e}')

    # ==================== Markers ====================

    def _build_container_markers(self):
        markers = []
        wall_color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.2)

        def make_wall(marker_id, cx, cy, cz, sx, sy, sz):
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.ns = NS_CONTAINER
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = Pose()
            m.pose.position = Point(x=float(cx), y=float(cy), z=float(cz))
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=float(sx), y=float(sy), z=float(sz))
            m.color = wall_color
            return m

        # Floor: spans X and Y, sits just below z=0
        markers.append(make_wall(
            0,
            self.L / 2.0, self.W / 2.0, -self.T / 2.0,
            self.L, self.W, self.T))

        # Back wall: at x=0, spans Y and Z
        markers.append(make_wall(
            1,
            -self.T / 2.0, self.W / 2.0, self.H / 2.0,
            self.T, self.W, self.H))

        # Left wall: at y=0, spans X and Z
        markers.append(make_wall(
            2,
            self.L / 2.0, -self.T / 2.0, self.H / 2.0,
            self.L, self.T, self.H))

        # Right wall: at y=W, spans X and Z
        markers.append(make_wall(
            3,
            self.L / 2.0, self.W + self.T / 2.0, self.H / 2.0,
            self.L, self.T, self.H))

        return markers

    def _color_from_id(self, package_id):
        h = hashlib.md5(package_id.encode()).digest()
        return ColorRGBA(r=float(h[0] / 255.0),
                         g=float(h[1] / 255.0),
                         b=float(h[2] / 255.0),
                         a=0.9)

    def _add_box_marker(self, package_id, corner_x, corner_y, corner_z,
                        length, width, height):
        m = Marker()
        m.header.frame_id = FRAME_ID
        m.ns = NS_PACKAGES
        m.id = int(hashlib.md5(package_id.encode()).hexdigest()[:8], 16) % 100000
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose = Pose()
        m.pose.position.x = float(corner_x) + float(length) / 2.0
        m.pose.position.y = float(corner_y) + float(width) / 2.0
        m.pose.position.z = float(corner_z) + float(height) / 2.0
        m.pose.orientation.w = 1.0

        m.scale = Vector3(x=float(length), y=float(width), z=float(height))
        m.color = self._color_from_id(package_id)

        self.box_markers[package_id] = m

    def _publish_all(self):
        msg = MarkerArray()
        msg.markers = self.container_markers + list(self.box_markers.values())
        now = self.get_clock().now().to_msg()
        for m in msg.markers:
            m.header.stamp = now
        self.marker_pub.publish(msg)

    # ==================== Callbacks ====================

    def on_placement_received(self, msg: PlacementTarget):
        response = self.get_state_client.call(GetState.Request())
        for i, pkg in enumerate(response.packages):
            if pkg.id == msg.package_id:
                pose = response.poses[i]
                self._add_box_marker(
                    package_id=pkg.id,
                    corner_x=pose.position.x,
                    corner_y=pose.position.y,
                    corner_z=pose.position.z,
                    length=pkg.length,
                    width=pkg.width,
                    height=pkg.height,
                )
                self._publish_all()
                self.get_logger().info(
                    f'Added marker for {pkg.id} at '
                    f'({pose.position.x:.2f}, {pose.position.y:.2f}, '
                    f'{pose.position.z:.2f})'
                )
                return

        self.get_logger().warn(
            f'Placement received for {msg.package_id} but it was not found '
            f'in container_state; no marker added'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
