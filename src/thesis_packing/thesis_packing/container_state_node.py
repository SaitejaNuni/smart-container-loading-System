"""
Container State Node — with real collision check + real support fraction.

Owns the container state: voxel grid, placed boxes, extreme points.
Exposes four services that other nodes call.

Also subscribes to /package_placed events to automatically update state
when the physical robot completes a placement.

Current state (all real):
  - PlaceBox: updates box list, voxel grid, extreme points
  - EvaluateCandidate: AABB collision + voxel-based support fraction
  - GetState: returns box list
  - GetCandidatePositions: sorted + container-fit filtered

Parameters:
  container_length (float, default 1.0): container X dimension (m)
  container_width (float, default 1.8): container Y dimension (m)
  container_height (float, default 1.8): container Z dimension (m)
  voxel_size (float, default 0.01): voxel edge length (m)
  auto_place_on_bridge_event (bool, default True): subscribe to /package_placed
    and automatically call PlaceBox when the bridge reports a physical placement
"""

import json
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String
from thesis_msgs.msg import PackageInfo
from thesis_msgs.srv import (
    EvaluateCandidate,
    PlaceBox,
    GetState,
    GetCandidatePositions,
)


# Tolerance for floating-point boundary comparisons.
# Prevents false collisions when boxes conceptually touch at exact boundaries
# but drift slightly due to float representation of 0.1, 0.2, 0.3, etc.
EPSILON = 1e-6


class ContainerStateNode(Node):
    """Maintains container state and serves queries about it."""

    def __init__(self):
        super().__init__('container_state')

        # ---- Parameters ----
        self.declare_parameter('container_length', 1.0)
        self.declare_parameter('container_width', 1.8)
        self.declare_parameter('container_height', 1.8)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('auto_place_on_bridge_event', True)

        self.container_length = self.get_parameter('container_length').value
        self.container_width = self.get_parameter('container_width').value
        self.container_height = self.get_parameter('container_height').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.auto_place_on_bridge_event = self.get_parameter(
            'auto_place_on_bridge_event').value

        # ---- Voxel grid ----
        self.nx = int(self.container_length / self.voxel_size)
        self.ny = int(self.container_width / self.voxel_size)
        self.nz = int(self.container_height / self.voxel_size)
        self.voxel_grid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        # ---- Box list ----
        self.placed_boxes = []

        # ---- Extreme points ----
        self.extreme_points = [(0.0, 0.0, 0.0)]

        # ---- Service servers ----
        self.evaluate_srv = self.create_service(
            EvaluateCandidate, '/evaluate_candidate', self.on_evaluate_candidate
        )
        self.place_srv = self.create_service(
            PlaceBox, '/place_box', self.on_place_box
        )
        self.get_state_srv = self.create_service(
            GetState, '/get_state', self.on_get_state
        )
        self.get_candidates_srv = self.create_service(
            GetCandidatePositions, '/get_candidate_positions', self.on_get_candidates
        )

        # ---- Bridge integration (subscribe to physical placement events) ----
        if self.auto_place_on_bridge_event:
            self.placement_sub = self.create_subscription(
                String, '/package_placed', self._on_bridge_placement, 10
            )
            self.get_logger().info(
                'Subscribed to /package_placed for auto state updates'
            )

        self.get_logger().info(
            f'Container state node ready. '
            f'Container: {self.container_length}x{self.container_width}x{self.container_height}m, '
            f'Grid: {self.nx}x{self.ny}x{self.nz} voxels at {self.voxel_size}m'
        )

    # ==================== Bridge event handler ====================

    def _on_bridge_placement(self, msg):
        """Update state when a placement event arrives.

        Message format (JSON string):
          {
            "name": "package_07",
            "position_m": [x, y, z],           # CENTER of package
            "orientation_quat_wxyz": [w, x, y, z],
            "dimensions_m": [dx, dy, dz],      # world-axis extents
            "mass_kg": M,
            "event": "placed"
          }
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f'Malformed /package_placed message: {e}')
            return

        if data.get('event') != 'placed':
            return

        class _R:
            pass
        req = _R()
        req.package_id = data['name']
        pos = data['position_m']
        dims = data['dimensions_m']
        # position_m is the CENTER of the package in the container frame.
        # PlaceBox uses back-left-bottom corner. Convert and clamp tiny
        # negatives that arise from float rounding (center - dims/2).
        req.x = max(0.0, pos[0] - dims[0] / 2.0)
        req.y = max(0.0, pos[1] - dims[1] / 2.0)
        req.z = max(0.0, pos[2] - dims[2] / 2.0)
        req.length = dims[0]
        req.width = dims[1]
        req.height = dims[2]
        req.mass = data['mass_kg']

        class _Resp:
            success = False
            error_message = ''
        resp = _Resp()
        self.on_place_box(req, resp)

        if resp.success:
            self.get_logger().info(
                f'Auto-placed {req.package_id} from /package_placed'
            )
        else:
            self.get_logger().warn(
                f'Failed to auto-place {req.package_id}: {resp.error_message}'
            )

    # ==================== Helpers ====================

    def world_to_voxel_indices(self, x, y, z, length, width, height):
        ix_min = int(x / self.voxel_size)
        iy_min = int(y / self.voxel_size)
        iz_min = int(z / self.voxel_size)
        ix_max = int((x + length) / self.voxel_size)
        iy_max = int((y + width) / self.voxel_size)
        iz_max = int((z + height) / self.voxel_size)
        return ix_min, iy_min, iz_min, ix_max, iy_max, iz_max

    def fits_in_container(self, x, y, z, length, width, height):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + length > self.container_length:
            return False
        if y + width > self.container_width:
            return False
        if z + height > self.container_height:
            return False
        return True

    def _is_point_inside_box(self, point, box_request):
        x, y, z = point
        return (
            box_request.x <= x < box_request.x + box_request.length and
            box_request.y <= y < box_request.y + box_request.width and
            box_request.z <= z < box_request.z + box_request.height
        )

    def _is_point_inside_any_box(self, point):
        x, y, z = point
        for box in self.placed_boxes:
            if (box['x'] <= x < box['x'] + box['length'] and
                box['y'] <= y < box['y'] + box['width'] and
                box['z'] <= z < box['z'] + box['height']):
                return True
        return False

    # ==================== Service handlers ====================

    def on_evaluate_candidate(self, request, response):
        self.get_logger().info(
            f'EvaluateCandidate called: '
            f'pos=({request.x:.2f}, {request.y:.2f}, {request.z:.2f}), '
            f'size=({request.length:.2f}, {request.width:.2f}, {request.height:.2f})'
        )

        if not self.fits_in_container(
            request.x, request.y, request.z,
            request.length, request.width, request.height
        ):
            response.collides = True
            response.support_fraction = 0.0
            self.get_logger().info('  -> out of container bounds (treated as collision)')
            return response

        cand_x_max = request.x + request.length
        cand_y_max = request.y + request.width
        cand_z_max = request.z + request.height

        response.collides = False
        colliding_id = None
        for box in self.placed_boxes:
            box_x_max = box['x'] + box['length']
            box_y_max = box['y'] + box['width']
            box_z_max = box['z'] + box['height']

            if (request.x >= box_x_max - EPSILON or box['x'] >= cand_x_max - EPSILON or
                request.y >= box_y_max - EPSILON or box['y'] >= cand_y_max - EPSILON or
                request.z >= box_z_max - EPSILON or box['z'] >= cand_z_max - EPSILON):
                continue

            response.collides = True
            colliding_id = box['id']
            break

        if response.collides:
            self.get_logger().info(f'  -> COLLIDES with {colliding_id}')
            response.support_fraction = 0.0
            return response

        ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = self.world_to_voxel_indices(
            request.x, request.y, request.z,
            request.length, request.width, request.height
        )

        if iz_min == 0:
            response.support_fraction = 1.0
        else:
            support_layer = self.voxel_grid[ix_min:ix_max, iy_min:iy_max, iz_min - 1]
            supported_voxels = int(support_layer.sum())
            total_voxels = int(support_layer.size)
            if total_voxels == 0:
                response.support_fraction = 0.0
            else:
                response.support_fraction = float(supported_voxels) / float(total_voxels)

        self.get_logger().info(
            f'  -> no collision, support_fraction={response.support_fraction:.3f}'
        )

        return response

    def on_place_box(self, request, response):
        self.get_logger().info(
            f'PlaceBox called: {request.package_id} at '
            f'({request.x:.2f}, {request.y:.2f}, {request.z:.2f})'
        )

        if not self.fits_in_container(
            request.x, request.y, request.z,
            request.length, request.width, request.height
        ):
            response.success = False
            response.error_message = 'Box would extend past container walls'
            return response

        self.placed_boxes.append({
            'id': request.package_id,
            'x': request.x,
            'y': request.y,
            'z': request.z,
            'length': request.length,
            'width': request.width,
            'height': request.height,
            'mass': request.mass,
        })

        ix_min, iy_min, iz_min, ix_max, iy_max, iz_max = self.world_to_voxel_indices(
            request.x, request.y, request.z,
            request.length, request.width, request.height
        )
        self.voxel_grid[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] = True

        # Update extreme points
        self.extreme_points = [
            ep for ep in self.extreme_points
            if not self._is_point_inside_box(ep, request)
        ]
        self.extreme_points.append((request.x + request.length, request.y, request.z))
        self.extreme_points.append((request.x, request.y + request.width, request.z))
        self.extreme_points.append((request.x, request.y, request.z + request.height))
        self.extreme_points = [
            ep for ep in self.extreme_points
            if not self._is_point_inside_any_box(ep)
        ]
        self.extreme_points = list(set(self.extreme_points))

        self.get_logger().info(
            f'Placed {request.package_id}. Total boxes: {len(self.placed_boxes)}, '
            f'Extreme points: {len(self.extreme_points)}'
        )

        response.success = True
        response.error_message = ''
        return response

    def on_get_state(self, request, response):
        self.get_logger().info(
            f'GetState called. Placed boxes: {len(self.placed_boxes)}'
        )

        for box in self.placed_boxes:
            pkg = PackageInfo()
            pkg.id = box['id']
            pkg.length = box['length']
            pkg.width = box['width']
            pkg.height = box['height']
            pkg.mass = box['mass']
            response.packages.append(pkg)

            pose = Pose()
            pose.position.x = float(box['x'])
            pose.position.y = float(box['y'])
            pose.position.z = float(box['z'])
            pose.orientation.w = 1.0
            response.poses.append(pose)

        return response

    def on_get_candidates(self, request, response):
        self.get_logger().info(
            f'GetCandidatePositions called for box '
            f'({request.length:.2f}, {request.width:.2f}, {request.height:.2f})'
        )

        valid_eps = [
            ep for ep in self.extreme_points
            if self.fits_in_container(
                ep[0], ep[1], ep[2],
                request.length, request.width, request.height
            )
        ]

        sorted_eps = sorted(valid_eps, key=lambda p: (p[0], p[2], p[1]))

        for (x, y, z) in sorted_eps:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)
            response.candidates.append(point)

        self.get_logger().info(
            f'  -> returned {len(sorted_eps)} valid candidates '
            f'(from {len(self.extreme_points)} total EPs)'
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ContainerStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
