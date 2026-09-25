#!/usr/bin/env python3
"""
Interactive arm with grasp and release.

Commands:
    joints a b c d e f        set joints directly
    pos x y z                 move to position, keep orientation
    pose x y z qx qy qz qw    move to full pose
    down x y z                move, tool pointing down
    home                      go to home pose
    fk                        print current TCP pose
    js                        print current joint values
    spawn [i]                 spawn package i at pickup station
    grasp [i]                 move above package i and activate weld
    release                   deactivate the active weld
    exit                      quit
"""

import queue
import threading
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

# Home: inverted L, facing container
HOME_Q = np.array([3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw

# Grasp target offsets
GRASP_SITE_DX = 0.1     # site is 0.1 further in +X than wrist origin
GRASP_SITE_DZ = 0.23    # site is 0.23 above the package center
GRASP_SITE_QUAT_XYZW = np.array([0.0, -0.7071, 0.0, -0.7071])


class InteractiveArm:
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
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self.joint_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.joint_ids
        ]
        self.actuator_for_joint = {}
        for aid in range(self.model.nu):
            jid = int(self.model.actuator_trnid[aid, 0])
            self.actuator_for_joint[jid] = aid

        # Package lookups
        self.weld_ids = []
        self.package_qpos_adr = []
        self.package_qvel_adr = []
        self.package_body_ids = []
        for i in range(50):
            wid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                f"package_{i:02d}_weld",
            )
            self.weld_ids.append(wid)
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"package_{i:02d}"
            )
            self.package_body_ids.append(bid)
            jadr = self.model.body_jntadr[bid]
            self.package_qpos_adr.append(self.model.jnt_qposadr[jadr])
            self.package_qvel_adr.append(self.model.jnt_dofadr[jadr])

        self.ctrl_target = HOME_Q.copy()
        self.active_weld_idx = None
        self.command_queue = queue.Queue()
        self.running = True

        mujoco.mj_resetData(self.model, self.data)
        self.set_home()

    def set_home(self):
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = HOME_Q[i]
        self.ctrl_target = HOME_Q.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def apply_ctrl_target(self):
        for i, jid in enumerate(self.joint_ids):
            aid = self.actuator_for_joint.get(jid, -1)
            if aid >= 0:
                self.data.ctrl[aid] = self.ctrl_target[i]

    def get_site_pose(self):
        pos = self.data.site_xpos[self.site_id].copy()
        mat = self.data.site_xmat[self.site_id].copy()
        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, mat)
        return pos, q_wxyz

    def print_achieved(self, label=""):
        pos, q_wxyz = self.get_site_pose()
        q_xyzw = np.roll(q_wxyz, -1)
        print(f"  [{label}] pos:  "
              f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")
        print(f"  [{label}] quat: "
              f"({q_xyzw[0]:+.4f}, {q_xyzw[1]:+.4f}, "
              f"{q_xyzw[2]:+.4f}, {q_xyzw[3]:+.4f})")

    def print_joints(self, label=""):
        vals = [self.data.qpos[adr] for adr in self.joint_qpos_adr]
        print(f"  [{label}] joints: "
              f"{[f'{v:+.3f}' for v in vals]}")

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

    def _solve_ik_warm(self, pos, quat_xyzw, max_iters=200):
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

    def plan_joint_path(self, target_pos, target_quat_xyzw, n_waypoints=40):
        start_pos, start_q_wxyz = self.get_site_pose()
        end_pos = np.asarray(target_pos, dtype=float)
        end_q_wxyz = np.roll(np.asarray(target_quat_xyzw, dtype=float), 1)

        start_joints = np.array([
            self.data.qpos[adr] for adr in self.joint_qpos_adr
        ])
        joint_seq = []
        current = start_joints.copy()
        n_failed = 0

        for k in range(n_waypoints + 1):
            t = k / n_waypoints
            wp_pos = (1 - t) * start_pos + t * end_pos
            wp_q_wxyz = self.slerp(start_q_wxyz, end_q_wxyz, t)
            wp_q_xyzw = np.roll(wp_q_wxyz, -1)

            for i, adr in enumerate(self.joint_qpos_adr):
                self.configuration.q[adr] = current[i]

            ok = self._solve_ik_warm(wp_pos, wp_q_xyzw, max_iters=200)
            if not ok:
                n_failed += 1
                joint_seq.append(current.copy())
                continue
            current = np.array([
                self.configuration.q[adr] for adr in self.joint_qpos_adr
            ])
            joint_seq.append(current.copy())

        # Final solve
        for i, adr in enumerate(self.joint_qpos_adr):
            self.configuration.q[adr] = current[i]
        ok_final = self._solve_ik_warm(end_pos, np.roll(end_q_wxyz, -1),
                                       max_iters=1000)
        final_joints = np.array([
            self.configuration.q[adr] for adr in self.joint_qpos_adr
        ])
        joint_seq.append(final_joints)

        print(f"  waypoints: {len(joint_seq)}  "
              f"failed intermediate: {n_failed}  "
              f"final converged: {ok_final}")

        # Restore physics state
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = start_joints[i]
        mujoco.mj_forward(self.model, self.data)
        return joint_seq

    def play_joint_seq(self, joint_seq, viewer, steps_per_segment=60):
        for i in range(len(joint_seq) - 1):
            qa = joint_seq[i]
            qb = joint_seq[i + 1]
            for s in range(steps_per_segment):
                a = s / steps_per_segment
                q_now = (1 - a) * qa + a * qb
                for j, adr in enumerate(self.joint_qpos_adr):
                    self.data.qpos[adr] = q_now[j]
                self.ctrl_target = q_now
                self.apply_ctrl_target()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.0005)
        # Lock at final
        self.ctrl_target = joint_seq[-1].copy()
        for j, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = joint_seq[-1][j]
        for _ in range(500):
            self.apply_ctrl_target()
            mujoco.mj_step(self.model, self.data)
            viewer.sync()

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

    def spawn_at_pickup(self, idx):
        pickup_top = np.array([1.4, 0.0, 0.75])
        pos = pickup_top + np.array([0.0, 0.0, 0.25])
        self.spawn_package(idx, pos)
        return pos

    def get_package_position(self, idx):
        adr = self.package_qpos_adr[idx]
        return np.array(self.data.qpos[adr:adr + 3])

    def command_listener(self):
        print("=" * 60)
        print("Commands:")
        print("  joints a b c d e f        set joints directly")
        print("  pos x y z                 move, keep orientation")
        print("  pose x y z qx qy qz qw    move to full pose")
        print("  down x y z                move, tool pointing down")
        print("  home                      go to home")
        print("  fk                        print current TCP pose")
        print("  js                        print current joint values")
        print("  spawn [i]                 spawn package i at pickup")
        print("  grasp [i]                 move above package i and weld")
        print("  release                   deactivate active weld")
        print("  exit                      quit")
        print("=" * 60)
        print()

        while self.running:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.command_queue.put(("exit", None))
                break
            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()
            try:
                if cmd in ("exit", "quit"):
                    self.command_queue.put(("exit", None))
                    return
                elif cmd == "home":
                    self.command_queue.put(("home", None))
                elif cmd == "fk":
                    self.command_queue.put(("fk", None))
                elif cmd == "js":
                    self.command_queue.put(("js", None))
                elif cmd == "release":
                    self.command_queue.put(("release", None))
                elif cmd == "spawn":
                    idx = int(parts[1]) if len(parts) > 1 else 0
                    self.command_queue.put(("spawn", idx))
                elif cmd == "grasp":
                    idx = int(parts[1]) if len(parts) > 1 else 0
                    self.command_queue.put(("grasp", idx))
                elif cmd == "joints":
                    if len(parts) != 7:
                        print("  usage: joints a b c d e f")
                        continue
                    self.command_queue.put(
                        ("joints", [float(x) for x in parts[1:7]])
                    )
                elif cmd == "pos":
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    _, cur_q = self.get_site_pose()
                    self.command_queue.put(
                        ("pose", (xyz, np.roll(cur_q, -1).tolist()))
                    )
                elif cmd == "down":
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    self.command_queue.put(
                        ("pose", (xyz, TOP_DOWN_QUAT.tolist()))
                    )
                elif cmd == "pose":
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    quat = [float(parts[4]), float(parts[5]),
                            float(parts[6]), float(parts[7])]
                    self.command_queue.put(("pose", (xyz, quat)))
                else:
                    print(f"  unknown command: {cmd}")
            except (ValueError, IndexError):
                print("  could not parse numbers")

    def run(self):
        threading.Thread(
            target=self.command_listener, daemon=True,
        ).start()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -30
            viewer.cam.azimuth = 90
            viewer.cam.lookat[:] = [0.7, 0.0, 0.8]

            print("Initial pose:")
            self.print_joints("start")
            self.print_achieved("start")
            print()

            while viewer.is_running() and self.running:
                try:
                    cmd, arg = self.command_queue.get_nowait()
                except queue.Empty:
                    cmd, arg = None, None

                if cmd == "exit":
                    self.running = False
                    break
                if cmd == "fk":
                    self.print_achieved("fk")
                    continue
                if cmd == "js":
                    self.print_joints("js")
                    continue

                if cmd == "release":
                    if self.active_weld_idx is None:
                        print("[RELEASE] no active weld")
                        continue
                    wid = self.weld_ids[self.active_weld_idx]
                    self.data.eq_active[wid] = 0
                    print(f"[RELEASE] weld deactivated for "
                          f"package {self.active_weld_idx}")
                    self.active_weld_idx = None
                    # Let the package fall
                    for _ in range(500):
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                    continue

                if cmd == "spawn":
                    idx = arg
                    spawn_pos = self.spawn_at_pickup(idx)
                    print(f"[SPAWN] package {idx} at "
                          f"({spawn_pos[0]:+.3f}, {spawn_pos[1]:+.3f}, "
                          f"{spawn_pos[2]:+.3f})")
                    for _ in range(1500):
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.0005)
                    final = self.get_package_position(idx)
                    print(f"  settled at ({final[0]:+.4f}, "
                          f"{final[1]:+.4f}, {final[2]:+.4f})")
                    continue

                if cmd == "grasp":
                    idx = arg
                    pkg_pos = self.get_package_position(idx)
                    print(f"[GRASP] package {idx} at "
                          f"({pkg_pos[0]:+.4f}, {pkg_pos[1]:+.4f}, "
                          f"{pkg_pos[2]:+.4f})")
                    target_site_pos = pkg_pos + np.array(
                        [GRASP_SITE_DX, 0.0, GRASP_SITE_DZ]
                    )
                    print(f"  target site pos: "
                          f"({target_site_pos[0]:+.4f}, "
                          f"{target_site_pos[1]:+.4f}, "
                          f"{target_site_pos[2]:+.4f})")
                    joint_seq = self.plan_joint_path(
                        target_site_pos, GRASP_SITE_QUAT_XYZW
                    )
                    self.play_joint_seq(joint_seq, viewer)
                    self.print_achieved("after")

                    # Activate weld
                    wid = self.weld_ids[idx]
                    self.data.eq_active[wid] = 1
                    self.active_weld_idx = idx
                    print(f"  weld activated for package {idx}")
                    # Let physics settle the constraint
                    for _ in range(500):
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                    pkg_after = self.get_package_position(idx)
                    print(f"  package now at ({pkg_after[0]:+.4f}, "
                          f"{pkg_after[1]:+.4f}, {pkg_after[2]:+.4f})")
                    continue

                if cmd == "joints":
                    vals = np.array(arg, dtype=float)
                    print(f"[JOINTS] {[f'{v:+.3f}' for v in vals]}")
                    start = np.array([
                        self.data.qpos[adr] for adr in self.joint_qpos_adr
                    ])
                    for i in range(1000):
                        t = i / 1000
                        s = t * t * (3 - 2 * t)
                        self.ctrl_target = start + s * (vals - start)
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.001)
                    self.ctrl_target = vals.copy()
                    for _ in range(500):
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                    self.print_joints("after")
                    continue

                if cmd == "home":
                    print(f"[HOME] moving to home")
                    start = np.array([
                        self.data.qpos[adr] for adr in self.joint_qpos_adr
                    ])
                    for i in range(1000):
                        t = i / 1000
                        s = t * t * (3 - 2 * t)
                        self.ctrl_target = start + s * (HOME_Q - start)
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.001)
                    self.ctrl_target = HOME_Q.copy()
                    for _ in range(500):
                        self.apply_ctrl_target()
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                    self.print_joints("home")
                    continue

                if cmd == "pose":
                    xyz, quat = arg
                    print(f"[MOVE] pos=({xyz[0]:+.4f}, "
                          f"{xyz[1]:+.4f}, {xyz[2]:+.4f})")
                    joint_seq = self.plan_joint_path(xyz, quat)
                    self.play_joint_seq(joint_seq, viewer)
                    self.print_achieved("after")
                    final_pos, final_q = self.get_site_pose()
                    dpos = np.linalg.norm(np.array(xyz) - final_pos)
                    dot = abs(np.dot(np.roll(np.array(quat), 1), final_q))
                    dot = min(1.0, max(-1.0, dot))
                    dang = np.degrees(2.0 * np.arccos(dot))
                    print(f"  position error:    {dpos * 1000:.2f} mm")
                    print(f"  orientation error: {dang:.2f} deg")
                    continue

                self.apply_ctrl_target()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)

            self.running = False
        print("Exiting cleanly.")


def main():
    print(f"Loading: {MJCF_PATH}")
    arm = InteractiveArm(MJCF_PATH)
    arm.run()


if __name__ == "__main__":
    main()
