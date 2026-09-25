"""
Placement Planner Node — X-first wall building with orientation-aware output.
Serves /plan_placement service. Orchestrator calls with a PackageInfo,
receives back a PlacementTarget containing:
  - pose.position: geometric CENTER of the placed package (world frame)
  - pose.orientation: rotation quaternion for chosen orientation
  - dim_x/y/z: package extents along world X/Y/Z after orientation
Algorithm per package:
  1. Get candidates from state node (extreme points cover X-columns)
  2. Group by X value, sort ascending
  3. Try each X in order (smallest first)
  4. For each candidate, evaluate all 6 orientations, keep best
  5. First X yielding a valid placement wins
Hard filters (per orientation):
  - Package must fit within container walls
  - No collision with placed packages
  - support_fraction >= 0.7
  - Heavy packages only on floor or on other heavy packages
Scoring: weight distribution (0.35), stability (0.25), compactness (0.20),
CoG height (0.20). Total = 1.00.
Placement_planner does NOT call PlaceBox. Container_state auto-updates
from /package_placed events from the MuJoCo bridge after physical
placement completes.
Parameters:
  container_length (float, default 1.0): container X dimension (m)
  container_width (float, default 1.8): container Y dimension (m)
  container_height (float, default 1.8): container Z dimension (m)
  heavy_mass_threshold (float, default 7.5): mass above which "heavy" rule applies (kg)
"""
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
from thesis_msgs.msg import PackageInfo, PlacementTarget
from thesis_msgs.srv import (
    EvaluateCandidate,
    GetState,
    GetCandidatePositions,
    PlanPlacement,
)
# ---- Algorithm-level constants ----
EPSILON = 1e-6
POSITION_TOLERANCE = 0.01
MIN_SUPPORT_FRACTION = 0.7
# Scoring weights (sum = 1.00)
W_WEIGHT_DIST = 0.35
W_STABILITY = 0.25
W_COMPACTNESS = 0.20
W_COG = 0.20
# 6 axis-aligned orientations. Each entry:
#   (perm_func, quaternion_xyzw)
# perm_func(l, w, h) returns the package's dimensions along (X, Y, Z) in world.
# quaternion_xyzw is the rotation from default (l→X, w→Y, h→Z) to placed.
ORIENTATIONS = [
    (lambda l, w, h: (l, w, h),   (0.0,     0.0,     0.0,     1.0)),      # 0: identity
    (lambda l, w, h: (w, l, h),   (0.0,     0.0,     0.7071,  0.7071)),   # 1: 90° about +Z
    (lambda l, w, h: (l, h, w),   (0.7071,  0.0,     0.0,     0.7071)),   # 2: 90° about +X
    (lambda l, w, h: (h, w, l),   (0.0,     0.7071,  0.0,     0.7071)),   # 3: 90° about +Y
    (lambda l, w, h: (w, h, l),   (-0.5,    -0.5,    -0.5,    0.5)),      # 4: 120° about (-1,-1,-1)
    (lambda l, w, h: (h, l, w),   (0.5,     0.5,     0.5,     0.5)),      # 5: 120° about (1,1,1)
]
class PlacementPlanner(Node):
    def __init__(self):
        super().__init__('placement_planner')
        # ---- Parameters ----
        self.declare_parameter('container_length', 1.0)
        self.declare_parameter('container_width', 1.8)
        self.declare_parameter('container_height', 1.8)
        self.declare_parameter('heavy_mass_threshold', 7.5)
        self.container_length = self.get_parameter('container_length').value
        self.container_width = self.get_parameter('container_width').value
        self.container_height = self.get_parameter('container_height').value
        self.heavy_mass_threshold = self.get_parameter('heavy_mass_threshold').value
        self.cb_group = ReentrantCallbackGroup()
        # ---- Service clients (query container_state) ----
        self.get_candidates_client = self.create_client(
            GetCandidatePositions, '/get_candidate_positions',
            callback_group=self.cb_group)
        self.evaluate_client = self.create_client(
            EvaluateCandidate, '/evaluate_candidate',
            callback_group=self.cb_group)
        self.get_state_client = self.create_client(
            GetState, '/get_state',
            callback_group=self.cb_group)
        # ---- Service server (orchestrator calls this) ----
        self.plan_srv = self.create_service(
            PlanPlacement, '/plan_placement',
            self.on_plan_placement,
            callback_group=self.cb_group)
        # ---- Publisher (for logging/RViz) ----
        self.publisher = self.create_publisher(
            PlacementTarget, '/placement_target', 10)
        self._wait_for_services()
        self.get_logger().info(
            f'Placement planner ready. Container: '
            f'{self.container_length}x{self.container_width}x{self.container_height}m, '
            f'heavy_threshold={self.heavy_mass_threshold}kg, '
            f'min_support={MIN_SUPPORT_FRACTION}. '
            f'Serving /plan_placement.'
        )
    def _wait_for_services(self):
        for client, name in [
            (self.get_candidates_client, '/get_candidate_positions'),
            (self.evaluate_client, '/evaluate_candidate'),
            (self.get_state_client, '/get_state'),
        ]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f'Waiting for {name}...')
    # ==================== Service handler ====================
    def on_plan_placement(self, request, response):
        pkg = request.package
        target = self.process_package(pkg)
        if target is None:
            response.success = False
            response.reason = 'No valid placement found'
            return response
        response.success = True
        response.target = target
        response.reason = ''
        return response
    # ==================== Heavy-on-heavy check ====================
    def _is_heavy_on_heavy_ok(self, x, y, z, dim_x, dim_y, dim_z, mass, placed_boxes):
        if mass <= self.heavy_mass_threshold:
            return True
        if abs(z) < EPSILON:
            return True
        cand_x_max = x + dim_x
        cand_y_max = y + dim_y
        for box in placed_boxes:
            box_top = box['z'] + box['dim_z']
            if abs(box_top - z) > EPSILON:
                continue
            box_x_max = box['x'] + box['dim_x']
            box_y_max = box['y'] + box['dim_y']
            x_overlap = max(0.0, min(cand_x_max, box_x_max) - max(x, box['x']))
            y_overlap = max(0.0, min(cand_y_max, box_y_max) - max(y, box['y']))
            if x_overlap * y_overlap > 0:
                if box['mass'] <= self.heavy_mass_threshold:
                    return False
        return True
    # ==================== Scoring ====================
    def _compute_cog_score(self, z, dim_z):
        center_z = z + dim_z / 2.0
        return 1.0 - (float(center_z) / self.container_height)
    def _compute_stability_score(self, eval_response):
        return float(eval_response.support_fraction)
    def _compute_compactness_score(self, x, y, z, dim_x, dim_y, dim_z, placed_boxes):
        total_surface = 2.0 * (dim_x * dim_y + dim_x * dim_z + dim_y * dim_z)
        if total_surface == 0:
            return 0.0
        contact = 0.0
        if abs(x) < EPSILON: contact += dim_y * dim_z
        if abs(y) < EPSILON: contact += dim_x * dim_z
        if abs(z) < EPSILON: contact += dim_x * dim_y
        cand_x_max = x + dim_x
        cand_y_max = y + dim_y
        cand_z_max = z + dim_z
        for box in placed_boxes:
            bx, by, bz = box['x'], box['y'], box['z']
            bdx, bdy, bdz = box['dim_x'], box['dim_y'], box['dim_z']
            bx_max = bx + bdx
            by_max = by + bdy
            bz_max = bz + bdz
            if abs(cand_x_max - bx) < EPSILON:
                y_ov = max(0, min(cand_y_max, by_max) - max(y, by))
                z_ov = max(0, min(cand_z_max, bz_max) - max(z, bz))
                contact += y_ov * z_ov
            if abs(x - bx_max) < EPSILON:
                y_ov = max(0, min(cand_y_max, by_max) - max(y, by))
                z_ov = max(0, min(cand_z_max, bz_max) - max(z, bz))
                contact += y_ov * z_ov
            if abs(cand_y_max - by) < EPSILON:
                x_ov = max(0, min(cand_x_max, bx_max) - max(x, bx))
                z_ov = max(0, min(cand_z_max, bz_max) - max(z, bz))
                contact += x_ov * z_ov
            if abs(y - by_max) < EPSILON:
                x_ov = max(0, min(cand_x_max, bx_max) - max(x, bx))
                z_ov = max(0, min(cand_z_max, bz_max) - max(z, bz))
                contact += x_ov * z_ov
            if abs(cand_z_max - bz) < EPSILON:
                x_ov = max(0, min(cand_x_max, bx_max) - max(x, bx))
                y_ov = max(0, min(cand_y_max, by_max) - max(y, by))
                contact += x_ov * y_ov
            if abs(z - bz_max) < EPSILON:
                x_ov = max(0, min(cand_x_max, bx_max) - max(x, bx))
                y_ov = max(0, min(cand_y_max, by_max) - max(y, by))
                contact += x_ov * y_ov
        return min(1.0, contact / total_surface)
    def _compute_weight_score(self, msg, x, y, z, dim_x, dim_y, dim_z, placed_boxes):
        if abs(z) < EPSILON:
            return 1.0
        cand_x_max = x + dim_x
        cand_y_max = y + dim_y
        total_contact_area = 0.0
        weighted_mass_sum = 0.0
        for box in placed_boxes:
            box_top = box['z'] + box['dim_z']
            if abs(box_top - z) > EPSILON:
                continue
            box_x_max = box['x'] + box['dim_x']
            box_y_max = box['y'] + box['dim_y']
            x_overlap = max(0.0, min(cand_x_max, box_x_max) - max(x, box['x']))
            y_overlap = max(0.0, min(cand_y_max, box_y_max) - max(y, box['y']))
            contact_area = x_overlap * y_overlap
            if contact_area <= 0:
                continue
            total_contact_area += contact_area
            weighted_mass_sum += box['mass'] * contact_area
        if total_contact_area <= 0:
            return 0.0
        avg_mass_below = weighted_mass_sum / total_contact_area
        if msg.mass <= avg_mass_below:
            return 1.0
        return avg_mass_below / msg.mass
    def _compute_total_score(self, weight, stability, compactness, cog):
        return (W_WEIGHT_DIST * weight +
                W_STABILITY * stability +
                W_COMPACTNESS * compactness +
                W_COG * cog)
    # ==================== Placement search ====================
    def _find_best_placement_at_x(self, msg, x_candidates, placed_boxes):
        """Try all candidates at this X, all 6 orientations. Return the best or None.
        
        best = (candidate, orientation_idx, (dim_x, dim_y, dim_z), total, ...)
        where dim_x/y/z are the package's extents along world axes for the
        chosen orientation.
        """
        best = None
        for candidate in x_candidates:
            for orientation_idx, (perm_func, quaternion) in enumerate(ORIENTATIONS):
                dim_x, dim_y, dim_z = perm_func(msg.length, msg.width, msg.height)
                if candidate.x + dim_x > self.container_length: continue
                if candidate.y + dim_y > self.container_width: continue
                if candidate.z + dim_z > self.container_height: continue
                eval_req = EvaluateCandidate.Request()
                eval_req.x = candidate.x
                eval_req.y = candidate.y
                eval_req.z = candidate.z
                eval_req.length = dim_x
                eval_req.width = dim_y
                eval_req.height = dim_z
                eval_response = self.evaluate_client.call(eval_req)
                if eval_response.collides:
                    continue
                if eval_response.support_fraction < MIN_SUPPORT_FRACTION:
                    continue
                if not self._is_heavy_on_heavy_ok(
                    candidate.x, candidate.y, candidate.z, dim_x, dim_y, dim_z,
                    msg.mass, placed_boxes
                ):
                    continue
                stability = self._compute_stability_score(eval_response)
                compactness = self._compute_compactness_score(
                    candidate.x, candidate.y, candidate.z,
                    dim_x, dim_y, dim_z, placed_boxes)
                weight_score = self._compute_weight_score(
                    msg, candidate.x, candidate.y, candidate.z,
                    dim_x, dim_y, dim_z, placed_boxes)
                cog = self._compute_cog_score(candidate.z, dim_z)
                total = self._compute_total_score(
                    weight_score, stability, compactness, cog)
                if best is None or total > best[3]:
                    best = (candidate, orientation_idx, (dim_x, dim_y, dim_z), total,
                            weight_score, stability, compactness, cog)
        return best
    def _group_by_x(self, candidates):
        sorted_cands = sorted(candidates, key=lambda c: c.x)
        groups = []
        if not sorted_cands:
            return groups
        current_x = sorted_cands[0].x
        current_group = []
        for cand in sorted_cands:
            if abs(cand.x - current_x) < POSITION_TOLERANCE:
                current_group.append(cand)
            else:
                groups.append((current_x, current_group))
                current_x = cand.x
                current_group = [cand]
        if current_group:
            groups.append((current_x, current_group))
        return groups
    # ==================== Package processing ====================
    def process_package(self, msg: PackageInfo):
        """Compute a placement for the given package.
        
        Returns a PlacementTarget on success, None on failure.
        """
        self.get_logger().info(
            f'Planning {msg.id}: '
            f'{msg.length:.2f} x {msg.width:.2f} x {msg.height:.2f} m, '
            f'{msg.mass:.2f} kg'
        )
        state_response = self.get_state_client.call(GetState.Request())
        placed_boxes = []
        for i, pkg in enumerate(state_response.packages):
            pose = state_response.poses[i]
            placed_boxes.append({
                'id': pkg.id,
                'x': pose.position.x, 'y': pose.position.y, 'z': pose.position.z,
                'dim_x': pkg.length, 'dim_y': pkg.width, 'dim_z': pkg.height,
                'mass': pkg.mass,
            })
        cand_req = GetCandidatePositions.Request()
        cand_req.length = msg.length
        cand_req.width = msg.width
        cand_req.height = msg.height
        cand_response = self.get_candidates_client.call(cand_req)
        candidates = list(cand_response.candidates)
        if not candidates:
            self.get_logger().warn(f'  No candidates returned for {msg.id}')
            return None
        x_groups = self._group_by_x(candidates)
        best = None
        chosen_x = None
        for x_value, x_candidates in x_groups:
            best = self._find_best_placement_at_x(msg, x_candidates, placed_boxes)
            if best is not None:
                chosen_x = x_value
                break
        if best is None:
            self.get_logger().warn(
                f'  No valid placement for {msg.id} '
                f'(tried {len(x_groups)} X-columns)')
            return None
        (candidate, orientation_idx, oriented_dims, total,
         weight_score, stability, compactness, cog) = best
        dim_x, dim_y, dim_z = oriented_dims
        _, chosen_quat = ORIENTATIONS[orientation_idx]
        # Compute package CENTER in world frame
        # Extreme point is the corner; center = corner + half_dims
        center_x = candidate.x + dim_x / 2.0
        center_y = candidate.y + dim_y / 2.0
        center_z = candidate.z + dim_z / 2.0
        self.get_logger().info(
            f'  Chose orientation={orientation_idx} '
            f'corner=({candidate.x:.2f}, {candidate.y:.2f}, {candidate.z:.2f}) '
            f'dims_in_world=({dim_x:.2f}, {dim_y:.2f}, {dim_z:.2f}) '
            f'center=({center_x:.2f}, {center_y:.2f}, {center_z:.2f}) '
            f'X-col={chosen_x:.2f} '
            f'score={total:.3f} '
            f'(w={weight_score:.2f} s={stability:.2f} c={compactness:.2f} '
            f'cog={cog:.2f})'
        )
        # Build PlacementTarget
        target = PlacementTarget()
        target.package_id = msg.id
        target.pose = Pose()
        target.pose.position.x = float(center_x)
        target.pose.position.y = float(center_y)
        target.pose.position.z = float(center_z)
        target.pose.orientation.x = float(chosen_quat[0])
        target.pose.orientation.y = float(chosen_quat[1])
        target.pose.orientation.z = float(chosen_quat[2])
        target.pose.orientation.w = float(chosen_quat[3])
        target.dim_x = float(dim_x)
        target.dim_y = float(dim_y)
        target.dim_z = float(dim_z)
        target.score = float(total)
        self.publisher.publish(target)
        return target
def main(args=None):
    rclpy.init(args=args)
    node = PlacementPlanner()
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
