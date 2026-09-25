import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    pkg_share_path = get_package_share_directory("ur10_mujoco_bringup")

    # Generate URDF via xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        os.path.join(pkg_share_path, "urdf", "ur10_mujoco.urdf.xacro"),
        " name:=ur ur_type:=ur10e",
    ])
    robot_description_str = robot_description_content.perform(context)

    # Save URDF to a temporary file for MoveItConfigsBuilder
    urdf_path = "/tmp/ur10_mujoco_generated.urdf"
    with open(urdf_path, "w") as f:
        f.write(robot_description_str)

    # Build the base MoveIt config from the UR vendor package
    moveit_config = (
        MoveItConfigsBuilder("ur10", package_name="ur_moveit_config")
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(
            file_path=os.path.join(pkg_share_path, "config", "ur10.srdf")
        )
        .robot_description_kinematics(
            file_path=os.path.join(pkg_share_path, "config", "kinematics.yaml")
        )
        .joint_limits(
            file_path=os.path.join(pkg_share_path, "config", "joint_limits.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(pkg_share_path, "config", "moveit_controllers.yaml")
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Convert to dict so we can override the OMPL planner config
    moveit_params = moveit_config.to_dict()

    # Replace the empty UR OMPL config with our own that declares RRTstar etc.
    # MoveIt stores the OMPL config under the key "ompl" at the top level of
    # the params tree for the move_group node.
    moveit_params["ompl"] = {
        "planning_plugins": ["ompl_interface/OMPLPlanner"],
        "request_adapters": [
            "default_planning_request_adapters/ResolveConstraintFrames",
            "default_planning_request_adapters/ValidateWorkspaceBounds",
            "default_planning_request_adapters/CheckStartStateBounds",
            "default_planning_request_adapters/CheckStartStateCollision",
        ],
        "response_adapters": [
            "default_planning_response_adapters/AddTimeOptimalParameterization",
            "default_planning_response_adapters/ValidateSolution",
            "default_planning_response_adapters/DisplayMotionPath",
        ],
        "planner_configs": {
            "RRTConnect": {
                "type": "geometric::RRTConnect",
                "range": 0.0,
            },
            "RRTstar": {
                "type": "geometric::RRTstar",
                "range": 0.0,
                "goal_bias": 0.05,
                "delay_collision_checking": 1,
            },
            "RRT": {
                "type": "geometric::RRT",
                "range": 0.0,
                "goal_bias": 0.05,
            },
            "PRM": {
                "type": "geometric::PRM",
                "max_nearest_neighbors": 10,
            },
        },
        "ur_manipulator": {
            "default_planner_config": "RRTstar",
            "planner_configs": ["RRTConnect", "RRTstar", "RRT", "PRM"],
            "projection_evaluator": "joints(shoulder_pan_joint,shoulder_lift_joint)",
            "longest_valid_segment_fraction": 0.05,
        },
    }

    parameters_file = os.path.join(pkg_share_path, "config", "controllers.yaml")

    nodes = []

    # robot_state_publisher
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[
                {"robot_description": robot_description_str},
                {"use_sim_time": True},
            ],
        )
    )

    # MuJoCo ros2_control node
    nodes.append(
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[{"use_sim_time": True}, parameters_file],
        )
    )

    # Controller spawners
    for controller in ["joint_state_broadcaster", "joint_trajectory_controller"]:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", parameters_file],
                output="both",
            )
        )

    # MoveIt2 move_group with our planner config
    moveit_params["use_sim_time"] = True
    nodes.append(
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="both",
            parameters=[moveit_params],
        )
    )

    # RViz2
    rviz_config = os.path.join(pkg_share_path, "config", "moveit.rviz")
    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            output="both",
            arguments=["-d", rviz_config],
            parameters=[
                {"robot_description": robot_description_str},
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                {"use_sim_time": True},
            ],
        )
    )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("ur_type", default_value="ur10e"),
            OpaqueFunction(function=launch_setup),
        ]
    )
