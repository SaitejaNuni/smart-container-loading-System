cmake_minimum_required(VERSION 3.8)
project(ur10_mujoco_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(mujoco_ros2_control REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(controller_manager REQUIRED)
find_package(joint_state_broadcaster REQUIRED)
find_package(joint_trajectory_controller REQUIRED)
find_package(xacro REQUIRED)
find_package(ur_description REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY urdf config launch mjcf mjcf_thesis
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/random_pose_node.py
  scripts/pick_place_orchestrator.py
  scripts/check_ik_visual.py
  scripts/check_container_frame.py
  scripts/check_reachability.py
  scripts/check_ik_accuracy.py
  scripts/check_ik_motion.py
  scripts/check_arm_actual.py
  scripts/check_pickup_only.py
  scripts/check_tool_orientation.py
  scripts/publish_container_to_moveit.py
  scripts/test_pickup_direct.py
  scripts/test_rotation_only.py
  scripts/diagnose_ssik_frame.py
  scripts/diagnose_weld.py
  scripts/diagnose_frames.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
