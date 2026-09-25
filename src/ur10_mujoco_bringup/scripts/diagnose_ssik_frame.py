#!/usr/bin/env python3
"""Check whether ssik's FK matches MuJoCo's FK for the same joint values."""

import numpy as np
import mujoco
from pathlib import Path
from ssik.prebuilt import ur10e_ik


MJCF = Path.home() / (
    "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
)

JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Joint configuration to test — HOME2
Q = np.array([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

# ssik info
print("ssik metadata:")
print(f"  BASE_LINK     = {ur10e_ik.BASE_LINK}")
print(f"  EE_LINK       = {ur10e_ik.EE_LINK}")
print(f"  DOF           = {ur10e_ik.DOF}")
print(f"  SOLVER_NAME   = {ur10e_ik.SOLVER_NAME}")
print(f"  T_HOME shape  = {np.asarray(ur10e_ik.T_HOME).shape}")
print(f"  T_HOME        =")
print(np.asarray(ur10e_ik.T_HOME))
print()

# ssik FK on Q
T_ssik = ur10e_ik.fk(Q)
print("ssik.fk(Q):")
print(T_ssik)
print(f"  pos  = ({T_ssik[0,3]:+.4f}, {T_ssik[1,3]:+.4f}, {T_ssik[2,3]:+.4f})")
print()

# MuJoCo FK on Q
model = mujoco.MjModel.from_xml_path(str(MJCF))
data = mujoco.MjData(model)
for i, name in enumerate(JOINTS):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = Q[i]
mujoco.mj_forward(model, data)

# Try every plausible tool frame
for site_or_body in ["attachment_site", "tool0", "wrist_3_link", "flange"]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_or_body)
    if sid >= 0:
        pos = data.site_xpos[sid]
        print(f"MuJoCo site '{site_or_body}' pos = "
              f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        continue
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, site_or_body)
    if bid >= 0:
        pos = data.xpos[bid]
        print(f"MuJoCo body '{site_or_body}' pos = "
              f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
