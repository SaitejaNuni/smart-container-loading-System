#!/usr/bin/env python3
"""Test weld geometry with the arm held at home."""

from pathlib import Path
import mujoco
import numpy as np

MJCF_PATH = Path.home() / (
    "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
)

HOME_Q = np.array([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in JOINT_NAMES
    ]
    actuator_for_joint = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        actuator_for_joint[jid] = aid

    # Set home + hold with ctrl
    for i, jid in enumerate(joint_ids):
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = HOME_Q[i]
    for i, jid in enumerate(joint_ids):
        aid = actuator_for_joint[jid]
        data.ctrl[aid] = HOME_Q[i]

    wid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "package_00_weld"
    )
    rel_pos = model.eq_data[wid, 3:6].copy()
    rel_quat = model.eq_data[wid, 6:10].copy()

    print("Weld relpose from model:")
    print(f"  pos  = ({rel_pos[0]:+.4f}, {rel_pos[1]:+.4f}, {rel_pos[2]:+.4f})")
    print(f"  quat (wxyz) = ({rel_quat[0]:+.4f}, {rel_quat[1]:+.4f}, "
          f"{rel_quat[2]:+.4f}, {rel_quat[3]:+.4f})")

    wrist_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
    )
    pkg_bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "package_00"
    )
    pkg_jadr = model.body_jntadr[pkg_bid]
    pkg_qadr = model.jnt_qposadr[pkg_jadr]

    def expected_pkg_pos():
        wp = data.xpos[wrist_bid].copy()
        wq = data.xquat[wrist_bid].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, wq)
        return wp + R.reshape(3, 3) @ rel_pos

    # Teleport package to expected position before activating weld
    expected = expected_pkg_pos()
    data.qpos[pkg_qadr + 0] = expected[0]
    data.qpos[pkg_qadr + 1] = expected[1]
    data.qpos[pkg_qadr + 2] = expected[2]
    data.qpos[pkg_qadr + 3] = 1.0
    data.qpos[pkg_qadr + 4] = 0.0
    data.qpos[pkg_qadr + 5] = 0.0
    data.qpos[pkg_qadr + 6] = 0.0

    mujoco.mj_forward(model, data)

    print()
    print("Before weld activation:")
    print(f"  wrist pos:   ({data.xpos[wrist_bid][0]:+.4f}, "
          f"{data.xpos[wrist_bid][1]:+.4f}, "
          f"{data.xpos[wrist_bid][2]:+.4f})")
    print(f"  pkg pos:     ({data.xpos[pkg_bid][0]:+.4f}, "
          f"{data.xpos[pkg_bid][1]:+.4f}, "
          f"{data.xpos[pkg_bid][2]:+.4f})")
    dx = data.xpos[pkg_bid][0] - data.xpos[wrist_bid][0]
    dy = data.xpos[pkg_bid][1] - data.xpos[wrist_bid][1]
    dz = data.xpos[pkg_bid][2] - data.xpos[wrist_bid][2]
    print(f"  pkg - wrist: ({dx:+.4f}, {dy:+.4f}, {dz:+.4f})")

    # Activate weld + step
    data.eq_active[wid] = 1
    for _ in range(500):
        for i, jid in enumerate(joint_ids):
            aid = actuator_for_joint[jid]
            data.ctrl[aid] = HOME_Q[i]
        mujoco.mj_step(model, data)

    print()
    print("After weld active + 500 steps (ctrl holds home):")
    print(f"  wrist pos:   ({data.xpos[wrist_bid][0]:+.4f}, "
          f"{data.xpos[wrist_bid][1]:+.4f}, "
          f"{data.xpos[wrist_bid][2]:+.4f})")
    print(f"  pkg pos:     ({data.xpos[pkg_bid][0]:+.4f}, "
          f"{data.xpos[pkg_bid][1]:+.4f}, "
          f"{data.xpos[pkg_bid][2]:+.4f})")
    dx = data.xpos[pkg_bid][0] - data.xpos[wrist_bid][0]
    dy = data.xpos[pkg_bid][1] - data.xpos[wrist_bid][1]
    dz = data.xpos[pkg_bid][2] - data.xpos[wrist_bid][2]
    print(f"  pkg - wrist: ({dx:+.4f}, {dy:+.4f}, {dz:+.4f})")
    print(f"  distance:    {np.sqrt(dx*dx+dy*dy+dz*dz):.4f} m")


if __name__ == "__main__":
    main()
