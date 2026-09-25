#!/usr/bin/env python3
"""Open the MuJoCo scene with the arm in an inverted-L pose."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


MJCF_PATH = Path.home() / (
    "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
)

# The inverted-L pose: upper arm horizontal, elbow bent 90 degrees so the
# forearm points straight up. This is the pose in the earlier logs where the
# arm faced the container.
INVERTED_L_Q = np.array([
    0.0000,   # shoulder_pan      -> arm rotated 180 deg (facing container)
   -1.5708,   # shoulder_lift     -> upper arm horizontal
    1.5708,   # elbow             -> forearm bent up 90 deg
   -1.5708,   # wrist_1           -> wrist continues
   -1.5708,   # wrist_2
    0.0000,   # wrist_3
])

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main():
    print(f"Loading: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)

    print(f"Model: nq={model.nq}, nu={model.nu}")

    # Set the six arm joints to the inverted-L pose
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"WARNING: joint '{name}' not found")
            continue
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = INVERTED_L_Q[i]
        print(f"  {name:25s} = {INVERTED_L_Q[i]:+.4f}")

    # Zero all velocities
    data.qvel[:] = 0.0

    # Compute derived quantities (body positions, etc.)
    mujoco.mj_forward(model, data)

    print("Opening viewer. Close the window or press Ctrl+C to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 90
        viewer.cam.lookat[:] = [0.7, 0.0, 0.8]

        # Do NOT step physics. Just render the current pose.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
