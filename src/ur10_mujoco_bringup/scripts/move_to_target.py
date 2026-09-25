#!/usr/bin/env python3
"""Open the scene, move the arm to a commanded position + orientation."""

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

# ----- Home pose: inverted L, facing container -----
HOME_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


# =====================================================================
# COMMAND THE TARGET HERE
# =====================================================================
# Position: (x, y, z) in MJCF world metres.
# Orientation: quaternion in ROS convention (x, y, z, w).
#   Top-down grasp:   [1.0, 0.0, 0.0, 0.0]
#   Identity (up):    [0.0, 0.0, 0.0, 1.0]
#
# Example 1: pickup station, above it
# TARGET_POS = [1.4, 0.0, 1.20]
# TARGET_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
#
# Example 2: container interior, mid height
# TARGET_POS = [0.5, 0.0, 0.60]
# TARGET_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
#
# Example 3: in front of the robot at base height
# TARGET_POS = [1.5, 0.0, 0.45]
# TARGET_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]

TARGET_POS = [1.4, 0.0, 1.20]
TARGET_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
# =====================================================================


class ArmController:
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(str(path))
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
            mink.VelocityLimit(
                self.model, {j: np.pi for j in JOINT_NAMES}
            ),
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

        # Start at home
        mujoco.mj_resetData(self.model, self.data)
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = HOME_Q[i]
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    # ------------------------------------------------------------------

    def get_site_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].copy()
        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, mat)
        return pos, q_wxyz

    @staticmethod
    def slerp(q0, q1, t):
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        dot = float(np.dot(q0, q1))
        if dot < 0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            r = q0 + t * (q1 - q0)
            return r / np.linalg.norm(r)
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta_0 * t
        q2 = q1 - q0 * dot
        q2 = q2 / np.linalg.norm(q2)
        return q0 * np.cos(theta) + q2 * np.sin(theta)

    def solve_ik_for_target(self, pos, quat_xyzw, max_iters=100):
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

    def move_to_pose(self, target_pos, target_quat_xyzw, viewer=None,
                     n_waypoints=40, steps_per_segment=10):
        start_pos, start_q_wxyz = self.get_site_pose()
        end_pos = np.asarray(target_pos, dtype=float)
        end_q_wxyz = np.roll(np.asarray(target_quat_xyzw, dtype=float), 1)

        joint_seq = []
        for k in range(n_waypoints + 1):
            t = k / n_waypoints
            wp_pos = (1 - t) * start_pos + t * end_pos
            wp_q_wxyz = self.slerp(start_q_wxyz, end_q_wxyz, t)
            wp_q_xyzw = np.roll(wp_q_wxyz, -1)

            ok = self.solve_ik_for_target(wp_pos, wp_q_xyzw)
            if not ok:
                print(f"  WARNING: IK did not converge at waypoint {k}/{n_waypoints}")

            q = np.array([
                self.configuration.q[adr] for adr in self.joint_qpos_adr
            ])
            joint_seq.append(q)

        # Report final error
        final_pos, final_q = self.get_site_pose()
        dpos = np.linalg.norm(end_pos - final_pos)
        dot = abs(np.dot(end_q_wxyz, final_q))
        dot = min(1.0, max(-1.0, dot))
        dang = np.degrees(2.0 * np.arccos(dot))
        print(f"  Final position error: {dpos * 1000:.2f} mm")
        print(f"  Final orientation error: {dang:.2f} deg")

        # Play back the joint trajectory
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

        for j, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = joint_seq[-1][j]
        mujoco.mj_forward(self.model, self.data)


def main():
    print(f"Loading: {MJCF_PATH}")
    arm = ArmController(MJCF_PATH)

    print()
    print("=" * 60)
    print(f"Commanded position:    {TARGET_POS}")
    print(f"Commanded orientation: {TARGET_QUAT_XYZW}  (xyzw)")
    print("=" * 60)

    with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
        mujoco.mjv_defaultFreeCamera(arm.model, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90
        viewer.cam.lookat[:] = [0.7, 0.0, 0.8]

        # Show home first
        print("Showing home pose for 2 seconds...")
        for _ in range(200):
            mujoco.mj_step(arm.model, arm.data)
            viewer.sync()
            time.sleep(0.001)

        print("Moving to target...")
        arm.move_to_pose(
            TARGET_POS, TARGET_QUAT_XYZW, viewer=viewer,
        )

        print("Hold. Close the viewer to exit.")
        while viewer.is_running():
            mujoco.mj_step(arm.model, arm.data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
