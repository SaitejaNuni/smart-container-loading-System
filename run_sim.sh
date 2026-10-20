#!/bin/bash
# Launch the full pick-and-place pipeline across six terminals.

WS=~/thesis_mujoco_ws
SOURCE="source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash"

# Terminal 1 — Simulation + MoveIt2
gnome-terminal --title="T1: Simulation" -- bash -c \
  "$SOURCE && cd $WS && ros2 launch ur10_mujoco_bringup ur10_mujoco_moveit.launch.py; exec bash"

sleep 10   # give the simulation time to load before starting the rest

# Terminal 2 — Container publisher
gnome-terminal --title="T2: Container Publisher" -- bash -c \
  "$SOURCE && ros2 run ur10_mujoco_bringup publish_container_to_moveit.py; exec bash"

sleep 1

# Terminal 3 — Generator
gnome-terminal --title="T3: Generator" -- bash -c \
  "$SOURCE && ros2 run thesis_packing random_package_generator_node; exec bash"

sleep 1

# Terminal 4 — Container state
gnome-terminal --title="T4: Container State" -- bash -c \
  "$SOURCE && ros2 run thesis_packing container_state_node; exec bash"

sleep 1

# Terminal 5 — Placement planner
gnome-terminal --title="T5: Placement Planner" -- bash -c \
  "$SOURCE && ros2 run thesis_packing placement_planner_node; exec bash"

sleep 1

# Terminal 6 — Driver (edit num_packages as needed)
gnome-terminal --title="T6: Driver" -- bash -c \
  "$SOURCE && ros2 run ur10_mujoco_bringup pick_place_driver.py --ros-args -p num_packages:=3; exec bash"
