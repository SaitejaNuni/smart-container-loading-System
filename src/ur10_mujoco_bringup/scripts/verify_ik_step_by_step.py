#!/usr/bin/env python3
"""Step-by-step IK verification.
For each target:
  - Print requested position + orientation
  - Solve IK (or accept preset joints)
  - Run FK on the result
  - Print achieved position + orientation
  - Print error in mm and degrees
  - Move the arm there (smooth interpolation)
  - Hold for 3 s so you can watch
"""

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

HOME_Q = np.array([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# -------- Test cases --------
# "type": "joints" means set joint angles directly (FK verifies them).
# "type": "pose" means run mink IK on the target pose.
TEST_CASES = [
    {
        "name": "1. HOME (reference joints)",
        "type": "joints",
        "joints": [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    },
    {
        "name": "2. PICKUP approach via ROS2-verified joints",
        "type": "joints",
        "joints": [-0.471, -1.945, 2.088, -0.143, 1.099, -1.571],
    },
    {
        "name": "3. PICKUP approach via mink IK (fresh)",
        "type": "pose",
        "position": [1.4, 0.0, 1.28],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
    },
    {
        "name": "4. Return to HOME",
        "type": "joints",
        "joints": [3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    },
    {
        "name": "5. PLACE target via mink IK",
        "type": "pose",
        "position": [0.5, 0.0, 0.58],
        "quat_xyzw": [1.0, 0.0, 0.0, 0.0],
    },
]


def find_ee_site(model):
    """Try several common site names, return the first that exists."""
    for name in ["attachment_site", "tool0", "tool_frame", "ee_site", "tcp"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            print(f"  EE site found: '{name}' (id={sid})")
            return name, sid
    # Fall back to wrist_3_link body
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    if bid >= 0:
        print(f"  No EE site; using body 'wrist_3_link' (id={bid})")
        return "wrist_3_link", bid
    raise RuntimeError("No EE site or wrist_3_link body found")


def get_ee_pose(data, site_name, site_id):
    """Return (pos, quat_wxyz) for the EE in world frame."""
    sid = mujoco.mj_name2id(
        data.model if hasattr(data, "model") else None,
        mujoco.mjtObj.mjOBJ_SITE, site_name,
    )
    # Use the passed id
    sid = site_id
    pos = data.site_xpos[sid].copy()
    xmat = data.site_xmat[sid].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, xmat)  # wxyz
    return pos, quat


def quat_error_deg(q1_wxyz, q2_wxyz):
    dot = abs(np.dot(q1_wxyz, q2_wxyz))
    dot = min(1.0, max(-1.0, dot))
    return np.degrees(2.0 * np.arccos(dot))


class Verifier:
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)

        self.site_name, self.site_id = find_ee_site(self.model)

        self.ee_task = mink.FrameTask(
            frame_name=self.site_name,
            frame_type="site" if self.site_name != "wrist_3_link" else "body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-5)
        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            mink.VelocityLimit(self.model, {j: np.pi for j in JOINT_NAMES}),
        ]

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in JOINT_NAMES
        ]
        self.joint_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.joint_ids
        ]

        mujoco.mj_resetData(self.model, self.data)
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = HOME_Q[i]
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def set_joints(self, joints):
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = joints[i]
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)

    def solve_ik(self, position, quat_xyzw, max_iters=300):
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        target = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=np.roll(quat_xyzw, 1)),
            translation=position,
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

    def apply_configuration(self, steps=600):
        target = np.array([
            self.configuration.q[adr] for adr in self.joint_qpos_adr
        ])
        start = np.array([
            self.data.qpos[adr] for adr in self.joint_qpos_adr
        ])
        for k in range(steps):
            t = k / max(steps - 1, 1)
            s = t * t * (3.0 - 2.0 * t)
            for i, adr in enumerate(self.joint_qpos_adr):
                self.data.qpos[adr] = start[i] + s * (target[i] - start[i])
            mujoco.mj_step(self.model, self.data)
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = target[i]
        mujoco.mj_forward(self.model, self.data)


def main():
    print(f"Loading: {MJCF_PATH}")
    v = Verifier(MJCF_PATH)

    with mujoco.viewer.launch_passive(v.model, v.data) as viewer:
        mujoco.mjv_defaultFreeCamera(v.model, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90
        viewer.cam.lookat[:] = [0.7, 0.0, 0.8]

        for case in TEST_CASES:
            print()
            print("=" * 70)
            print(case["name"])

            if case["type"] == "joints":
                joints = np.array(case["joints"])
                print(f"  Requested joints: "
                      f"{[f'{j:+.3f}' for j in joints]}")
                v.set_joints(joints)
                achieved_pos, achieved_quat = get_ee_pose(
                    v.data, v.site_name, v.site_id,
                )
                print(f"  Achieved pose (MJCF world):")
                print(f"    pos  = ({achieved_pos[0]:+.4f}, "
                      f"{achieved_pos[1]:+.4f}, {achieved_pos[2]:+.4f})")
                print(f"    quat = ({achieved_quat[1]:+.4f}, "
                      f"{achieved_quat[2]:+.4f}, {achieved_quat[3]:+.4f}, "
                      f"{achieved_quat[0]:+.4f})  # xyzw")
            else:
                pos = np.array(case["position"])
                quat = np.array(case["quat_xyzw"])
                print(f"  Requested position (MJCF world): "
                      f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
                print(f"  Requested quat xyzw:             "
                      f"({quat[0]:+.4f}, {quat[1]:+.4f}, "
                      f"{quat[2]:+.4f}, {quat[3]:+.4f})")

                ok = v.solve_ik(pos, quat)
                joints = np.array([
                    v.configuration.q[adr] for adr in v.joint_qpos_adr
                ])
                print(f"  IK solved: {ok}")
                print(f"  IK joints:  {[f'{j:+.3f}' for j in joints]}")

                # FK on the IK solution
                v.set_joints(joints)
                achieved_pos, achieved_quat = get_ee_pose(
                    v.data, v.site_name, v.site_id,
                )
                dpos = np.linalg.norm(achieved_pos - pos)
                dquat = quat_error_deg(achieved_quat, np.roll(quat, 1))
                print(f"  Achieved position: "
                      f"({achieved_pos[0]:+.4f}, {achieved_pos[1]:+.4f}, "
                      f"{achieved_pos[2]:+.4f})")
                print(f"  POSITION ERROR:    {dpos * 1000:.2f} mm")
                print(f"  ORIENTATION ERROR: {dquat:.2f} deg")

            # Move and hold
            v.apply_configuration(steps=600)
            for _ in range(200):
                mujoco.mj_step(v.model, v.data)
                viewer.sync()
                time.sleep(0.001)
            time.sleep(3.0)

        print()
        print("Done. Close the viewer to exit.")
        while viewer.is_running():
            mujoco.mj_step(v.model, v.data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
