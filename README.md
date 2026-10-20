# Thesis MuJoCo Workspace — UR10e Pick-and-Place

A ROS 2 + MuJoCo simulation of a UR10e industrial robot arm performing autonomous pick-and-place of randomized packages into a container. Motion planning uses MoveIt2, inverse kinematics uses an analytical UR10e solver, and grasping uses a custom C++ plugin that toggles MuJoCo weld constraints so the arm physically feels the payload.

This repository contains everything needed to build and run the full pipeline on a single Ubuntu machine.

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [Pipeline overview](#pipeline-overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Build](#build)
- [Running the demo](#running-the-demo)
- [The custom weld plugin](#the-custom-weld-plugin)
- [Placement planner](#placement-planner)
- [Metrics and logging](#metrics-and-logging)
- [Known limitations](#known-limitations)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Why this project exists

Testing a robot control architecture on real hardware is slow, expensive, and risky. Simulation lets us validate the full software stack — planning, kinematics, control, and physical grasping — long before the first physical trial.

The thesis asks two questions:

1. Can a full autonomous pick-and-place pipeline be assembled inside a modern ROS 2 + MuJoCo stack using off-the-shelf components?
2. Can packages be grasped in a way that is physically realistic enough that the arm actually feels the payload, without resorting to kinematic teleportation tricks?

The answer to both is yes, with one non-trivial piece of engineering: a custom C++ plugin that gives ROS 2 control over MuJoCo's weld equality constraints. Everything else is built from MoveIt2, `mujoco_ros2_control`, and a UR10e-specific analytical IK solver.

---

## Pipeline overview

The system consists of five ROS 2 nodes plus a compiled plugin, all coordinated by a test script that drives the sequence.
random_package_generator_node → produces random packages on demand
│
▼ PackageInfo
placement_planner_node → decides where in the container
│
│ queries GetCandidatePositions, EvaluateCandidate, GetState
▼
container_state_node → tracks voxels, placed boxes, extreme points
│
▼ PlacementTarget
pick_place_driver.py → drives spawn, pick, place; uses ssik + MoveIt2
│
▼ joint trajectories + weld commands
mujoco_ros2_control + weld_plugin → physics loop, weld toggling

text

The cycle per package:

1. Request a package from the generator.
2. Spawn it on the pickup station (teleport by the plugin).
3. Compute the analytical IK solution for the pickup pose.
4. Move the arm, engage the weld, lift.
5. Move to the container, ask the planner for a placement target.
6. Compute the analytical IK solution for the placement pose.
7. Move, release the weld, notify `container_state`.
8. Return home and repeat.

---

## Repository layout

| Package | Purpose |
|---|---|
| `thesis_msgs` | Custom messages (`PackageInfo`, `PlacementTarget`) and services (`GetNextPackage`, `PlanPlacement`, `GetState`, `PlaceBox`, `EvaluateCandidate`, `GetCandidatePositions`) |
| `thesis_packing` | Generator, container state, placement planner, visualization |
| `ur10_mujoco_bringup` | Launch files, MJCF scene, URDF, controller configs, test scripts |
| `mujoco_ros2_control` | Vendored upstream + custom `weld_plugin` for physical grasping |
| `mujoco_vendor` | Vendored MuJoCo binaries and headers |
| `pymoveit2` | Vendored Python MoveIt2 client |

Key files:

- `src/ur10_mujoco_bringup/scripts/pick_place_driver.py` — main test driver
- `src/mujoco_ros2_control/mujoco_ros2_control_plugins/src/weld_plugin.cpp` — custom plugin
- `src/thesis_packing/thesis_packing/placement_planner_node.py` — packing algorithm
- `src/thesis_packing/thesis_packing/container_state_node.py` — container model
- `src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml` — MuJoCo scene

---

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- MoveIt2
- Python 3.12 with `numpy`
- `mujoco_ros2_control` dependencies (vendored in this workspace)

Recommended: at least 8 GB of RAM.

---

## Build

```bash
cd ~/thesis_mujoco_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
The first build takes a few minutes because the custom weld_plugin and the mujoco_ros2_control hardware interface are compiled from source. Subsequent builds are incremental.

If you only change a Python node, no rebuild is needed thanks to --symlink-install. If you change a C++ file, .msg, .srv, or package.xml, rebuild that package:

bash
colcon build --symlink-install --packages-select <package_name>
source install/setup.bash
Running the demo
Open six terminals. In each one, first run:

bash
source /opt/ros/jazzy/setup.bash
source ~/thesis_mujoco_ws/install/setup.bash
Terminal 1 — Simulation (MuJoCo + MoveIt2)
bash
ros2 launch ur10_mujoco_bringup ur10_mujoco_moveit.launch.py
Wait until the log shows:

text
[weld_plugin]: WeldPlugin init: 50 welds mapped
If you don't see it, the plugin failed to load — check the log.

Terminal 2 — Container publisher
bash
ros2 run ur10_mujoco_bringup publish_container_to_moveit.py
Tells MoveIt2 about the container as a collision object.

Terminal 3 — Package generator
bash
ros2 run thesis_packing random_package_generator_node
Generates random packages on demand.

Terminal 4 — Container state
bash
ros2 run thesis_packing container_state_node
Maintains the voxel grid, placed box list, and extreme points. Updates when the test script publishes /package_placed.

Terminal 5 — Placement planner
bash
ros2 run thesis_packing placement_planner_node
Serves /plan_placement. Given a package, returns where to put it.

Terminal 6 — Pick-and-place driver
bash
ros2 run ur10_mujoco_bringup pick_place_driver.py --ros-args -p num_packages:=3














## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
