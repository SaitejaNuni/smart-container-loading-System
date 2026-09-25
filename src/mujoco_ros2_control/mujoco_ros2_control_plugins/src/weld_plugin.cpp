#include "mujoco_ros2_control_plugins/weld_plugin.hpp"

#include <algorithm>
#include <cmath>
#include <pluginlib/class_list_macros.hpp>

namespace weld_plugin
{

// Pickup station top surface, MJCF world
static constexpr double PICKUP_X = 1.4;
static constexpr double PICKUP_Y = 0.0;
static constexpr double PICKUP_TOP_Z = 0.75;

// Minimum Y offset from the wrist so the package never intersects
// the wrist geometry, regardless of package height.
static constexpr double MIN_RELPOSE_Y = 0.10;

bool WeldPlugin::init(rclcpp::Node::SharedPtr node, const mjModel * model, mjData * /*data*/)
{
  node_ = node;
  model_mutable_ = const_cast<mjModel *>(model);

  weld_ids_.assign(50, -1);
  pkg_qpos_adr_.assign(50, -1);
  pkg_qvel_adr_.assign(50, -1);
  pkg_body_id_.assign(50, -1);

  int found = 0;
  for (int i = 0; i < model->neq; ++i) {
    if (model->eq_type[i] != mjEQ_WELD) continue;
    const int body2_id = model->eq_obj2id[i];
    if (body2_id < 0) continue;
    const char * body2_name = mj_id2name(model, mjOBJ_BODY, body2_id);
    if (!body2_name) continue;
    std::string name(body2_name);
    if (name.rfind("package_", 0) != 0) continue;

    int idx;
    try { idx = std::stoi(name.substr(8)); } catch (...) { continue; }
    if (idx < 0 || idx >= 50) continue;

    weld_ids_[idx] = i;
    pkg_body_id_[idx] = body2_id;
    ++found;
    if (wrist_body_id_ < 0) wrist_body_id_ = model->eq_obj1id[i];

    const int jnt_adr = model->body_jntadr[body2_id];
    if (jnt_adr >= 0 && model->jnt_type[jnt_adr] == mjJNT_FREE) {
      pkg_qpos_adr_[idx] = model->jnt_qposadr[jnt_adr];
      pkg_qvel_adr_[idx] = model->jnt_dofadr[jnt_adr];
    }

    // Force a stiff, critically damped constraint for every weld so the
    // grasped package does not oscillate under its own weight.
    model_mutable_->eq_solref[i * 2 + 0] = 0.005;   // time constant (s)
    model_mutable_->eq_solref[i * 2 + 1] = 1.0;     // critically damped
    model_mutable_->eq_solimp[i * 5 + 0] = 0.99;    // dmin
    model_mutable_->eq_solimp[i * 5 + 1] = 0.999;   // dmax
    model_mutable_->eq_solimp[i * 5 + 2] = 0.0001;  // width
    model_mutable_->eq_solimp[i * 5 + 3] = 0.5;     // mid
    model_mutable_->eq_solimp[i * 5 + 4] = 2.0;     // power
  }

  RCLCPP_INFO(logger_, "WeldPlugin init: %d welds mapped", found);

  grip_sub_ = node_->create_subscription<std_msgs::msg::Bool>(
    "/gripper_command", 10,
    [this](const std_msgs::msg::Bool::SharedPtr msg) {
      grip_command_.store(msg->data);
    });

  active_pkg_sub_ = node_->create_subscription<std_msgs::msg::Int32>(
    "/active_package_index", 10,
    [this](const std_msgs::msg::Int32::SharedPtr msg) {
      active_package_id_.store(msg->data);
    });

  relpose_sub_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/weld_plugin/set_relpose", 10,
    [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      on_relpose_msg(msg);
    });

  spawn_service_ = node_->create_service<std_srvs::srv::Trigger>(
    "/spawn_package",
    [this](
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
      const int active = active_package_id_.load();
      if (active < 0) {
        res->success = false;
        res->message = "No active package index set";
        return;
      }
      pending_spawn_idx_.store(active);
      res->success = true;
      res->message = "Spawn requested for package " + std::to_string(active);
    });

  return true;
}

void WeldPlugin::on_relpose_msg(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 8) {
    RCLCPP_WARN(logger_, "set_relpose expects 8 doubles: idx x y z qw qx qy qz");
    return;
  }
  const int idx = static_cast<int>(msg->data[0]);
  if (idx < 0 || idx >= 50) return;
  if (weld_ids_[idx] < 0) return;

  RelposeUpdate u;
  u.weld_id = weld_ids_[idx];
  u.pos[0] = msg->data[1];
  // Clamp Y so the package cannot penetrate the wrist geometry.
  u.pos[1] = std::max(static_cast<double>(msg->data[2]), MIN_RELPOSE_Y);
  u.pos[2] = msg->data[3];
  u.quat[0] = msg->data[4];
  u.quat[1] = msg->data[5];
  u.quat[2] = msg->data[6];
  u.quat[3] = msg->data[7];

  std::lock_guard<std::mutex> lock(updates_mutex_);
  pending_updates_.push_back(u);
}

void WeldPlugin::pre_step(mjData * data)
{
  // 1. Apply pending relpose updates
  {
    std::lock_guard<std::mutex> lock(updates_mutex_);
    for (const auto & u : pending_updates_) {
      const int base = u.weld_id * 11;
      model_mutable_->eq_data[base + 3] = u.pos[0];
      model_mutable_->eq_data[base + 4] = u.pos[1];
      model_mutable_->eq_data[base + 5] = u.pos[2];
      model_mutable_->eq_data[base + 6] = u.quat[0];
      model_mutable_->eq_data[base + 7] = u.quat[1];
      model_mutable_->eq_data[base + 8] = u.quat[2];
      model_mutable_->eq_data[base + 9] = u.quat[3];

      RCLCPP_INFO(
        logger_,
        "weld %d relpose updated: pos=(%.4f, %.4f, %.4f) quat=(%.4f, %.4f, %.4f, %.4f)",
        u.weld_id, u.pos[0], u.pos[1], u.pos[2],
        u.quat[0], u.quat[1], u.quat[2], u.quat[3]);
    }
    pending_updates_.clear();
  }

  // 2. Handle spawn request
  const int spawn_idx = pending_spawn_idx_.exchange(-1);
  if (spawn_idx >= 0) {
    teleport_to_pickup(data, spawn_idx);
  }

  // 3. Read the relpose of the active package
  const int active = active_package_id_.load();
  if (active >= 0 && active < 50 && weld_ids_[active] >= 0) {
    const int base = weld_ids_[active] * 11;
    relpose_pos_[0] = model_mutable_->eq_data[base + 3];
    relpose_pos_[1] = model_mutable_->eq_data[base + 4];
    relpose_pos_[2] = model_mutable_->eq_data[base + 5];
    relpose_quat_[0] = model_mutable_->eq_data[base + 6];
    relpose_quat_[1] = model_mutable_->eq_data[base + 7];
    relpose_quat_[2] = model_mutable_->eq_data[base + 8];
    relpose_quat_[3] = model_mutable_->eq_data[base + 9];
  }

  // 4. Rising-edge teleport to wrist
  const bool grip = grip_command_.load();
  const bool rising_edge = grip && !prev_grip_;
  const bool target_changed =
    (active != prev_active_) && grip && active >= 0;

  if ((rising_edge || target_changed) && grip && active >= 0) {
    teleport_to_wrist(data, active);
  }

  // 5. Toggle welds
  for (int i = 0; i < static_cast<int>(weld_ids_.size()); ++i) {
    const int wid = weld_ids_[i];
    if (wid < 0) continue;
    const bool should_be_active = grip && (i == active);
    data->eq_active[wid] = should_be_active ? 1 : 0;
  }

  prev_grip_ = grip;
  prev_active_ = active;
}

void WeldPlugin::teleport_to_wrist(mjData * data, int pkg_idx)
{
  if (wrist_body_id_ < 0 || pkg_idx < 0 || pkg_idx >= 50) return;
  const int qpos_adr = pkg_qpos_adr_[pkg_idx];
  if (qpos_adr < 0) return;

  const double * wp = &data->xpos[wrist_body_id_ * 3];
  const double * wq = &data->xquat[wrist_body_id_ * 4];
  const double * wmat = &data->xmat[wrist_body_id_ * 9];

  // ---- Position --------------------------------------------------------
  double offset_world[3];
  mju_mulMatVec3(offset_world, wmat, relpose_pos_);

  data->qpos[qpos_adr + 0] = wp[0] + offset_world[0];
  data->qpos[qpos_adr + 1] = wp[1] + offset_world[1];
  data->qpos[qpos_adr + 2] = wp[2] + offset_world[2];

  double q_world[4];
  mju_mulQuat(q_world, wq, relpose_quat_);
  data->qpos[qpos_adr + 3] = q_world[0];
  data->qpos[qpos_adr + 4] = q_world[1];
  data->qpos[qpos_adr + 5] = q_world[2];
  data->qpos[qpos_adr + 6] = q_world[3];

  // ---- Velocity --------------------------------------------------------
  // Match the package's velocity to the wrist's velocity at the package's
  // new location. Without this, the weld engages with the package at rest
  // while the arm is in motion, producing a jerk and ringing.
  const int qvel_adr = pkg_qvel_adr_[pkg_idx];
  if (qvel_adr < 0) return;

  const double * wcvel = &data->cvel[wrist_body_id_ * 6];  // [ang(3), lin(3)]
  const double * w_com = &data->xipos[wrist_body_id_ * 3];

  // Vector from wrist CoM to the package's new position.
  const double r[3] = {
    data->qpos[qpos_adr + 0] - w_com[0],
    data->qpos[qpos_adr + 1] - w_com[1],
    data->qpos[qpos_adr + 2] - w_com[2],
  };

  // v_pkg = v_wrist_com + omega × r
  double omega_x_r[3];
  mju_cross(omega_x_r, &wcvel[0], r);

  data->qvel[qvel_adr + 0] = wcvel[3] + omega_x_r[0];
  data->qvel[qvel_adr + 1] = wcvel[4] + omega_x_r[1];
  data->qvel[qvel_adr + 2] = wcvel[5] + omega_x_r[2];

  // Angular velocity: match the wrist's angular velocity. MuJoCo stores
  // free-joint angular velocity in the global frame, which is what cvel
  // gives us directly.
  data->qvel[qvel_adr + 3] = wcvel[0];
  data->qvel[qvel_adr + 4] = wcvel[1];
  data->qvel[qvel_adr + 5] = wcvel[2];
}

void WeldPlugin::teleport_to_pickup(mjData * data, int pkg_idx)
{
  if (pkg_idx < 0 || pkg_idx >= 50) return;
  const int qpos_adr = pkg_qpos_adr_[pkg_idx];
  if (qpos_adr < 0) return;

  data->qpos[qpos_adr + 0] = PICKUP_X;
  data->qpos[qpos_adr + 1] = PICKUP_Y;
  data->qpos[qpos_adr + 2] = PICKUP_TOP_Z + 0.25;
  data->qpos[qpos_adr + 3] = 1.0;
  data->qpos[qpos_adr + 4] = 0.0;
  data->qpos[qpos_adr + 5] = 0.0;
  data->qpos[qpos_adr + 6] = 0.0;

  const int qvel_adr = pkg_qvel_adr_[pkg_idx];
  if (qvel_adr >= 0) {
    for (int k = 0; k < 6; ++k) data->qvel[qvel_adr + k] = 0.0;
  }
}

void WeldPlugin::update(const mjModel * /*model*/, mjData * /*data*/) {}

void WeldPlugin::on_reset(mjData * data)
{
  for (int wid : weld_ids_) {
    if (wid >= 0) data->eq_active[wid] = 0;
  }
  grip_command_.store(false);
  active_package_id_.store(-1);
  pending_spawn_idx_.store(-1);
  prev_grip_ = false;
  prev_active_ = -1;
  std::lock_guard<std::mutex> lock(updates_mutex_);
  pending_updates_.clear();
}

void WeldPlugin::cleanup() {}

}  // namespace weld_plugin

PLUGINLIB_EXPORT_CLASS(
  weld_plugin::WeldPlugin,
  mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
