//
// Created by brent on 2/21/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_SE2_TARGET_H
#define INFORMATION_GATHERING_LIBRARY_SE2_TARGET_H

#include <igl/target_models/target_model.h>
#include <igl/target_models/policies/control_policy.h>
#include <igl/se3_pose.h>
#include <igl/control/dd_motion_model.h>

namespace nx {

/**
 * @brief Models a Mobile Target in SE(2) driven by a Control Policy specified on construction.
 */
class SE2Target : public Target {

 protected:
  // System State
  SE3Pose pose;

  // System Model
  Matrix2d A = MatrixXd::Identity(2, 2); // Constant System Dynamics
  Matrix2d W = MatrixXd::Identity(2, 2); // Constant System Noise.

  // Sampling Period
  double tau {0.5};

  // Control Policy
  std::shared_ptr<ControlPolicy<3, 2>> controller;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct an SE(2) target.
   * @param ID The identification of the target.
   * @param se2_pose The initial pose of the target (X, Y, Yaw).
   * @param q The optional noise covariance parameter. Defaults to zero for a stationary target.
   */
  SE2Target(int ID, const Vector3d &se2_pose, std::shared_ptr<ControlPolicy<3, 2>> policy, double tau = 0.5, double q = 0) :
      Target(ID, 2), pose(se2_pose), W(q * MatrixXd::Identity(2, 2)), controller(policy) {}

  /**
   * @brief Return the 3D position of the target. Assumed to be in the z=0 plane.
   * @return The position as a 3-D vector.
   */
  Vector3d getPosition() const override {
    Vector3d position_3d;
    position_3d << pose.position[0], pose.position[1], 0;
    return position_3d;
  }

  /**
   * @brief Returns the state of interest about the target.
   * @return The target state being tracked.
   */
  VectorXd getState() const override {
    return pose.position.head(2);
  }

  /**
   * @brief Returns the Jacobian of the system dynamics about the current target state.
   * @return The Jacobian matrix.
   */
  MatrixXd getJacobian() const override {
    return A;
  }

  /**
   * @brief Returns the Noise matrix at the current target state.
   * @return The noise matrix.
   */
  MatrixXd getNoise() const override {
    return W;
  }

  /**
   * Updates the state of the Target by T steps, by simulating the gaussian dynamics.
   * @param T The horizon to simulate over.
   * @param map Map to restrict boundaries.
   * @param cmap Costmap to restrict collisions.
   */
  void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) override {
    for (int t = 0; t < T; t++) {      // Update the state T times.
      // Simulate Dynamics
      double x = pose.position[0];
      double y = pose.position[1];
      double th = pose.getYaw();
      Vector2d u = controller->computeControl(pose.position);
      std::array<double, 3> next_state = dd_motion_model({x, y , th}, {u[0], u[1]}, tau);
      Vector2d noise = normal_dist(W);
      pose.position[0] = next_state[0] + noise[0];
      pose.position[1] = next_state[1] + noise[1];
      pose.orientation = rotz(next_state[2]);
      restrict_position(map, cmap);
    }
  }

 protected:

  /**
   * @brief Restricts a 2D position variable to given map boundaries.
   * @param input The 2D position input.
   * @param map
   * @param cmap
   */
  void restrict_position(const nx::map_nd &map, const std::vector<char> &cmap) {
    double x_min = map.min()[0];
    double x_max = map.max()[0];
    double y_min = map.min()[1];
    double y_max = map.max()[1];

    auto clamp = [&](const double val, double min_value, double max_value) {
      if (val < min_value) {
        return min_value;
      } else if (val > max_value) {
        return max_value;
      } else {
        return val;
      }
    };
    // Bound the target state inside the valid map region.
    pose.position[0] = clamp(pose.position[0], x_min, x_max);
    pose.position[1] = clamp(pose.position[1], y_min, y_max);
  }

}; // End Class
} // End namespace.

#endif //INFORMATION_GATHERING_LIBRARY_SE2_TARGET_H
