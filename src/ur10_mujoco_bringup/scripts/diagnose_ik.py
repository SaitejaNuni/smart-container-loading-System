#!/usr/bin/env python3
"""Test whether mink's IK converges for the target pose."""

from pathlib import Path
import mujoco
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

TARGET_POS = np.array([1.4, 0.0, 1.2])
TARGET_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    config = mink.Configuration(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in JOINT_NAMES
    ]
    qpos_adr = [model.jnt_qposadr[j] for j in joint_ids]

    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = HOME_Q[i]
    mujoco.mj_forward(model, data)
    config.update(data.qpos)

    ee_task = mink.FrameTask(
        frame_name=EE_SITE, frame_type="site",
        position_cost=1.0, orientation_cost=1.0, lm_damping=1e-6,
    )
    posture_task = mink.PostureTask(model, cost=1e-5)
    posture_task.set_target_from_configuration(config)
    tasks = [ee_task, posture_task]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, {j: np.pi for j in JOINT_NAMES}),
    ]

    target = mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3(wxyz=np.roll(TARGET_QUAT_XYZW, 1)),
        translation=TARGET_POS,
    )
    ee_task.set_target(target)

    print(f"Target pos:  {TARGET_POS}")
    print(f"Target quat (xyzw): {TARGET_QUAT_XYZW}")
    print()

    for it in range(2000):
        vel = mink.solve_ik(
            config, tasks, dt=0.005, solver="daqp", limits=limits,
        )
        config.integrate_inplace(vel, dt=0.005)
        err = ee_task.compute_error(config)

        if it % 200 == 0 or it == 1999:
            print(f"  iter {it:4d}  "
                  f"pos err = {np.linalg.norm(err[:3]) * 1000:7.2f} mm  "
                  f"ori err = {np.linalg.norm(err[3:]):.4f}")

        if np.linalg.norm(err[:3]) < 0.001 and np.linalg.norm(err[3:]) < 0.01:
            print(f"  CONVERGED at iteration {it}")
            break

    final_q = np.array([config.q[adr] for adr in qpos_adr])
    print()
    print(f"Final joints: {[f'{v:+.3f}' for v in final_q]}")

    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = final_q[i]
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    final_pos = data.site_xpos[site_id].copy()
    print(f"FK position:  ({final_pos[0]:+.4f}, "
          f"{final_pos[1]:+.4f}, {final_pos[2]:+.4f})")
    dpos = np.linalg.norm(TARGET_POS - final_pos)
    print(f"POSITION ERROR: {dpos * 1000:.2f} mm")


if __name__ == "__main__":
    main()
