#!/usr/bin/env python3
"""Pure MuJoCo pick-and-place with Cartesian trajectory interpolation."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import mink


MJCF_PATH = Path.home() / (
    "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
)

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

EE_SITE = "attachment_site"
HOME_Q = np.array([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
PICKUP_TOP = np.array([1.4, 0.0, 0.75])
WELD_OFFSET_WRIST = np.array([0.23, 0.0, 0.0])
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class PureMujocoPickPlace:
    def __init__(self, mjcf_path):
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)

        self.ee_task = mink.FrameTask(
            frame_name=EE_SITE, frame_type="site",
            position_cost=1.0, orientation_cost=1.0, lm_damping=1e-6,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-5)
        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, {j: np.pi for j in JOINT_NAMES}),
        ]

        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE
        )

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in JOINT_NAMES
        ]
        self.joint_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.joint_ids
        ]

        # Package handling lookups
        self.weld_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                f"package_{i:02d}_weld",
            )
            for i in range(50)
        ]
        self.package_qpos_adr = []
        self.package_qvel_adr = []
        for i in range(50):
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{i:02d}"
            )
            jadr = self.model.body_jntadr[bid]
            self.package_qpos_adr.append(self.model.jnt_qposadr[jadr])
            self.package_qvel_adr.append(self.model.jnt_dofadr[jadr])

        self.active_package_index = 0
        self.gripping = False

        mujoco.mj_resetData(self.model, self.data)
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = HOME_Q[i]
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    # -------------------------------------------------------------------------

    def _get_site_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].copy()
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, mat)
        return pos, quat_wxyz

    @staticmethod
    def _slerp(q0, q1, t):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)
        return q0 * np.cos(theta) + q2 * np.sin(theta)

    def _solve_ik_for_target(self, pos, quat_xyzw, max_iters=80):
        """Set IK target and iterate. Returns True on convergence."""
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=np.roll(quat_xyzw, 1)),
            translation=np.asarray(pos, dtype=float),
        )
        self.ee_task.set_target(target)

        for _ in range(max_iters):
            vel = mink.solve_ik(
                self.configuration, self.tasks, dt=0.005,
                solver="daqp", limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, dt=0.005)
            err = self.ee_task.compute_error(self.configuration)
            if np.linalg.norm(err[:3]) < 0.002 and np.linalg.norm(err[3:]) < 0.02:
                return True
        return False

    def _play_joint_traj(self, joint_seq, steps_per_segment=15, viewer=None):
        """joint_seq: list of np.array(6). Play it smoothly through physics."""
        for i in range(len(joint_seq) - 1):
            qa = joint_seq[i]
            qb = joint_seq[i + 1]
            for s in range(steps_per_segment):
                a = s / steps_per_segment
                for j, adr in enumerate(self.joint_qpos_adr):
                    self.data.qpos[adr] = (1 - a) * qa[j] + a * qb[j]
                mujoco.mj_step(self.model, self.data)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.001)

        # Final position
        for j, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = joint_seq[-1][j]
        mujoco.mj_forward(self.model, self.data)

    def move_tool_to(self, target_pos, target_quat_xyzw,
                     n_waypoints=40, viewer=None):
        """Move the tool along a Cartesian straight line to the target.
        IK is solved at every waypoint, and the joints are played back."""
        start_pos, start_quat_wxyz = self._get_site_pose()
        end_pos = np.asarray(target_pos, dtype=float)
        end_quat_wxyz = np.roll(np.asarray(target_quat_xyzw, dtype=float), 1)

        joint_seq = []
        for k in range(n_waypoints + 1):
            t = k / n_waypoints
            wp_pos = (1 - t) * start_pos + t * end_pos
            wp_quat_wxyz = self._slerp(start_quat_wxyz, end_quat_wxyz, t)
            wp_quat_xyzw = np.roll(wp_quat_wxyz, -1)
            ok = self._solve_ik_for_target(wp_pos, wp_quat_xyzw)
            q = np.array([
                self.configuration.q[adr] for adr in self.joint_qpos_adr
            ])
            joint_seq.append(q)

        # Optional: print final error
        end_pos_actual, end_quat_actual = self._get_site_pose()
        dpos = np.linalg.norm(end_pos - end_pos_actual)
        dot = abs(np.dot(end_quat_wxyz, end_quat_actual))
        dot = min(1.0, max(-1.0, dot))
        dang = np.degrees(2.0 * np.arccos(dot))
        print(f"  final pos err: {dpos*1000:.2f} mm, "
              f"orient err: {dang:.2f} deg")

        self._play_joint_traj(joint_seq, steps_per_segment=10, viewer=viewer)

    # -------------------------------------------------------------------------

    def spawn_package(self, idx, pos):
        adr = self.package_qpos_adr[idx]
        self.data.qpos[adr + 0] = pos[0]
        self.data.qpos[adr + 1] = pos[1]
        self.data.qpos[adr + 2] = pos[2]
        self.data.qpos[adr + 3] = 1.0
        self.data.qpos[adr + 4] = 0.0
        self.data.qpos[adr + 5] = 0.0
        self.data.qpos[adr + 6] = 0.0
        vadr = self.package_qvel_adr[idx]
        for k in range(6):
            self.data.qvel[vadr + k] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def activate_weld(self, idx, active):
        wid = self.weld_ids[idx]
        if wid >= 0:
            self.data.eq_active[wid] = 1 if active else 0

    def get_wrist_pose(self):
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
        )
        pos = self.data.xpos[bid].copy()
        quat_wxyz = self.data.xquat[bid].copy()
        return pos, quat_wxyz

    # -------------------------------------------------------------------------
    # Main sequence
    # -------------------------------------------------------------------------

    def run(self):
        self.spawn_package(
            0,
            np.array([PICKUP_TOP[0], PICKUP_TOP[1], PICKUP_TOP[2] + 0.15]),
        )
        print("Spawned package 0 at pickup station")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = [0.7, 0.0, 0.8]

            # HOME
            print("[STATE] HOME")
            for i, adr in enumerate(self.joint_qpos_adr):
                self.data.qpos[adr] = HOME_Q[i]
            mujoco.mj_forward(self.model, self.data)
            self.configuration.update(self.data.qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
            for _ in range(300):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)

            # Approach pickup
            print("[STATE] APPROACH_PICKUP")
            pkg_center = np.array([
                PICKUP_TOP[0], PICKUP_TOP[1], PICKUP_TOP[2] + 0.15,
            ])
            approach_pos = pkg_center + np.array([0.0, 0.0, 0.23 + 0.20])
            self.move_tool_to(approach_pos, TOP_DOWN_QUAT, viewer=viewer)

            # Grasp (descend 20 cm)
            print("[STATE] GRASP")
            grasp_pos = pkg_center + np.array([0.0, 0.0, 0.23])
            self.move_tool_to(grasp_pos, TOP_DOWN_QUAT, viewer=viewer)

            # Attach
            wrist_pos, wrist_quat_wxyz = self.get_wrist_pose()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, wrist_quat_wxyz)
            offset_world = R.reshape(3, 3) @ WELD_OFFSET_WRIST
            target_pkg = wrist_pos + offset_world
            self.spawn_package(self.active_package_index, target_pkg)
            self.activate_weld(self.active_package_index, True)
            self.gripping = True
            print("  Weld activated")
            for _ in range(200):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)

            # Lift
            print("[STATE] LIFT")
            lift_pos = pkg_center + np.array([0.0, 0.0, 0.23 + 0.40])
            self.move_tool_to(lift_pos, TOP_DOWN_QUAT, viewer=viewer)

            # Approach place (container interior)
            print("[STATE] APPROACH_PLACE")
            place_center = np.array([0.5, 0.0, 0.20])
            place_pos = place_center + np.array([0.0, 0.0, 0.23 + 0.20])
            self.move_tool_to(place_pos, TOP_DOWN_QUAT, viewer=viewer)

            # Release
            print("[STATE] RELEASE")
            drop_pos = place_center + np.array([0.0, 0.0, 0.23])
            self.move_tool_to(drop_pos, TOP_DOWN_QUAT, viewer=viewer)
            self.activate_weld(self.active_package_index, False)
            self.gripping = False
            print("  Weld deactivated")
            for _ in range(500):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)

            # Retreat
            print("[STATE] RETREAT")
            self.move_tool_to(place_pos, TOP_DOWN_QUAT, viewer=viewer)

            print("Sequence complete. Close the viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)


def main():
    print(f"Loading MJCF: {MJCF_PATH}")
    if not MJCF_PATH.exists():
        print(f"ERROR: MJCF not found at {MJCF_PATH}")
        return
    app = PureMujocoPickPlace(MJCF_PATH)
    app.run()


if __name__ == "__main__":
    main()
