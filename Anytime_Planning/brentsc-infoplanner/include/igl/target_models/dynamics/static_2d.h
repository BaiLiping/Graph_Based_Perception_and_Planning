//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
#define INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
#include <igl/target_models/target_model.h>
#include <igl/target_models/policies/control_policy.h>

namespace nx {

/**
 * @brief Models a Static Target in 2 Dimensions driven by random Gaussian noise. May be used to model 2-D features
 * or moving targets to be tracked. A control policy can also be provided.
 */
class Static2D : public Target {

 protected:
  // System state (2D Position)
  Vector2d position;

  // System Model
  const Matrix2d A = MatrixXd::Identity(2, 2); // Constant System Dynamics
  const Matrix2d W;

  // Control Policy
  std::shared_ptr<ControlPolicy<2, 2>> controller;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a 2D target with static dynamics.
   * @param ID The identification of the target.
   * @param position The initial position of the target.
   * @param q The optional noise covariance parameter. Defaults to zero for a stationary target.
   */
  Static2D(int ID, const Vector2d &position, double q = 0) :
      Target(ID, 2), position(position), W(q * MatrixXd::Identity(2, 2)) {}

  /**
   * @brief Return the 3D position of the target. Assumed to be in the z=0 plane.
   * @return The position as a 3-D vector.
   */
  Vector3d getPosition() const override {
    Vector3d position_3d;
    position_3d << position[0], position[1], 0;
    return position_3d;
  }

  /**
   * @brief Returns the state of interest about the target.
   * @return The target state being tracked.
   */
  VectorXd getState() const override {
    return position;
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
      position = A * position + normal_dist(W);
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
    position[0] = clamp(position[0], x_min, x_max);
    position[1] = clamp(position[1], y_min, y_max);
  }

}; // End Class
} // End namespace.
#endif //INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
