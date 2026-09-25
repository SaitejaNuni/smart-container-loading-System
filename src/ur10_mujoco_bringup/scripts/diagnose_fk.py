#!/usr/bin/env python3
"""Compare mink FK vs MuJoCo FK for the same joint values."""

from pathlib import Path
import mujoco
import numpy as np
import mink


MJCF_PATH = Path.home() / (
    "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
)

EE_SITE = "attachment_site"

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Joint values from the last log
TEST_JOINTS = [3.328, -1.897, 1.741, -1.414, -1.597, -1.257]


def main():
    print(f"Loading: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    # List all sites
    print()
    print("Sites in the model:")
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        parent_body = model.site_bodyid[sid]
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, parent_body
        )
        print(f"  id={sid}  name='{name}'  body='{body_name}'")
        print(f"      site_pos (in body frame) = {model.site_pos[sid]}")
        print(f"      site_quat (wxyz) = {model.site_quat[sid]}")

    # Set joints
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = TEST_JOINTS[i]
    mujoco.mj_forward(model, data)

    # MuJoCo FK for the site
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    mujoco_pos = data.site_xpos[site_id].copy()
    mujoco_q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(mujoco_q_wxyz, data.site_xmat[site_id])

    print()
    print("Joint values set:")
    print(f"  {TEST_JOINTS}")
    print()
    print(f"MuJoCo FK for site '{EE_SITE}':")
    print(f"  pos  = ({mujoco_pos[0]:+.4f}, "
          f"{mujoco_pos[1]:+.4f}, {mujoco_pos[2]:+.4f})")
    print(f"  quat (xyzw) = ({mujoco_q_wxyz[1]:+.4f}, "
          f"{mujoco_q_wxyz[2]:+.4f}, {mujoco_q_wxyz[3]:+.4f}, "
          f"{mujoco_q_wxyz[0]:+.4f})")

    # mink FK for the same joints
    config = mink.Configuration(model)
    config.update(data.qpos)
    mink_transform = config.get_transform_frame_to_world(
        EE_SITE, "site"
    )
    mink_pos = mink_transform.translation()
    mink_quat_wxyz = mink_transform.rotation().wxyz

    print()
    print(f"mink FK for site '{EE_SITE}':")
    print(f"  pos  = ({mink_pos[0]:+.4f}, "
          f"{mink_pos[1]:+.4f}, {mink_pos[2]:+.4f})")
    print(f"  quat (xyzw) = ({mink_quat_wxyz[1]:+.4f}, "
          f"{mink_quat_wxyz[2]:+.4f}, {mink_quat_wxyz[3]:+.4f}, "
          f"{mink_quat_wxyz[0]:+.4f})")

    print()
    diff = np.linalg.norm(mujoco_pos - mink_pos)
    print(f"POSITION DIFFERENCE: {diff * 1000:.2f} mm")
    if diff > 0.001:
        print(">>> mink and MuJoCo disagree on where the site is <<<")
    else:
        print(">>> mink and MuJoCo agree <<<")


if __name__ == "__main__":
    main()
