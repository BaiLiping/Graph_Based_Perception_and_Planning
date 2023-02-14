//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_H
#define INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_H

#include <igl/target_models/target_model.h>

namespace nx {

/**
 * @brief Models a moving target in 2-D moving with double integrator dynamics. Can be used to track moving targets by
 * estimating velocity.
 */
class DoubleIntegrator2D : public Target {
 protected:

  // System state (2D Position)
  Vector2d position;
  Vector2d velocity;

  // System Model
  const Matrix4d A; // Double Integrator Dynamics
  const Matrix4d W; // Double Integrator Noise

  double max_velocity{1.0}; // Maximum Velocity
  double tau{0.5}; // Sampling Period

  // Control Policy
  std::shared_ptr<ControlPolicy < 2, 2>> controller;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a 2D target with double integrator dynamics, whose state is position and velocity.
   * @param ID The identification of the target.
   * @param position The initial position of the target.
   * @param velocity The initial velocity of the target.
   * @param tau The time discretization for the integrator.
   * @param max_velocity The maximum attainable velocity.
   * @param q The optional noise covariance parameter. Defaults to zero for a stationary target.
   */
  DoubleIntegrator2D(int ID, const Vector2d &position,
                     const Vector2d &velocity,
                     double tau,
                     double max_velocity = 1.0,
                     double q = 0) :
      Target(ID, 4), position(position), velocity(velocity),
      A(constructDynamics(tau)), W(constructNoise(tau, q)),
      max_velocity(max_velocity), tau(tau) {}

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
    Vector4d output;
    output << position, velocity;
    return output;
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
   */
  void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) {
    for (int t = 0; t < T; t++) {
      Vector4d state = A * getState() + normal_dist(W);
      position = state.head(2);
      velocity = state.tail(2);
      restrict_position(map, cmap);
    }

  }
 protected:

  void restrict_position(const nx::map_nd &map, const std::vector<char> &cmap) {

    double x_min = map.min()[0];
    double x_max = map.max()[0];
    double y_min = map.min()[1];
    double y_max = map.max()[1];

    auto in_bounds = [&](const Vector2d &state) {
      std::vector<int> cells = map.meters2cells({state[0], state[1]});
      char collision = cmap[map.subv2ind_colmajor({cells[0], cells[1]})];
      return ((state[0] >= x_min && state[0] <= x_max) && (state[1] >= y_min && state[1] <= y_max)
          && collision == '0');
    };

    auto clamp = [&](const double val, double min_value, double max_value) {
      if (val < min_value) {
        return min_value;
      } else if (val > max_value) {
        return max_value;
      } else {
        return val;
      }
    };

    if (!in_bounds(position)) {
      velocity[0] = -1 * velocity[0] / abs(velocity[0] + .01);
      velocity[1] = -1 * velocity[1] / abs(velocity[1] + .01);
    }

    // Bound the velocity by elements (can be changed)
    velocity[0] = clamp(velocity[0], -max_velocity, max_velocity);
    velocity[1] = clamp(velocity[1], -max_velocity, max_velocity);

    // Bound the target belief state (position only)
    position[0] = clamp(position[0], x_min, x_max);
    position[1] = clamp(position[1], y_min, y_max);

  }

  MatrixXd constructDynamics(double tau) {
    MatrixXd A_ = MatrixXd::Identity(4, 4);
    // Set Dynamics matrix
    A_.block(0, 2, 2, 2) = tau * MatrixXd::Identity(2, 2);
    return A_;
  }

  MatrixXd constructNoise(double tau, double q) {
    MatrixXd W_ = MatrixXd::Identity(4, 4);
    // Set Noise matrix
    W_.block(0, 0, 2, 2) = tau * tau * tau / 3 * MatrixXd::Identity(2, 2);
    W_.block(0, 2, 2, 2) = tau * tau / 2 * MatrixXd::Identity(2, 2);
    W_.block(2, 0, 2, 2) = tau * tau / 2 * MatrixXd::Identity(2, 2);
    W_.block(2, 2, 2, 2) = tau * MatrixXd::Identity(2, 2);
    W_ = W_ * q; // Apply diffusion strength

    return W_;
  }
}; // End Class
} // End Namespace
#endif //INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_H
