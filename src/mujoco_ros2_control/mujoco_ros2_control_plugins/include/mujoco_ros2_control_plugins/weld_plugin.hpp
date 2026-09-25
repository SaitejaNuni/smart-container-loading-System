#ifndef MUJOCO_ROS2_CONTROL_PLUGINS__WELD_PLUGIN_HPP_
#define MUJOCO_ROS2_CONTROL_PLUGINS__WELD_PLUGIN_HPP_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace weld_plugin
{

struct RelposeUpdate
{
  int weld_id;
  double pos[3];
  double quat[4];  // wxyz
};

class WeldPlugin : public mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase
{
public:
  WeldPlugin() = default;
  ~WeldPlugin() override = default;

  bool init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * data) override;
  void update(const mjModel * model, mjData * data) override;
  void pre_step(mjData * data) override;
  void on_reset(mjData * data) override;
  void cleanup() override;

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("weld_plugin")};

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr grip_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr active_pkg_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr relpose_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr spawn_service_;

  mjModel * model_mutable_{nullptr};

  std::vector<int> weld_ids_;
  std::vector<int> pkg_qpos_adr_;
  std::vector<int> pkg_qvel_adr_;
  std::vector<int> pkg_body_id_;
  int wrist_body_id_{-1};

  double relpose_pos_[3] = {0.0, 0.23, 0.0};
  double relpose_quat_[4] = {0.7071, 0.7071, 0.0, 0.0};

  std::atomic<bool> grip_command_{false};
  std::atomic<int> active_package_id_{-1};
  std::atomic<int> pending_spawn_idx_{-1};

  std::mutex updates_mutex_;
  std::vector<RelposeUpdate> pending_updates_;

  bool prev_grip_{false};
  int prev_active_{-1};

  void on_relpose_msg(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void teleport_to_wrist(mjData * data, int pkg_idx);
  void teleport_to_pickup(mjData * data, int pkg_idx);
};

}  // namespace weld_plugin

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__WELD_PLUGIN_HPP_
