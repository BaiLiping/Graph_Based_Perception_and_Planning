//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
#define INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
#include <igl/target_motion_models/target_model.h>

namespace nx {

/**
 * @brief Models a Static Target in 3 Dimensions driven by random Gaussian noise. May be used to model 3-D features
 * or moving targets to be tracked.
 */
class StaticTarget3D : public Target {

 private:

  // System state (3D Position)
  Vector3d position;

  // System Model
  const Matrix3d A = MatrixXd::Identity(3, 3); // Constant System Dynamics
  const Matrix3d W;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a 3D target with static dynamics.
   * @param ID The identification of the target.
   * @param position The initial position of the target.
   * @param q The optional noise covariance parameter. Defaults to zero for a stationary target.
   */
  StaticTarget3D(int ID, const Vector3d &position, double q = 0) :
      Target(ID, 3), position(position), W(q * MatrixXd::Identity(3, 3)) {}

  /**
   * @brief Return the 3D position of the target. Assumed to be in the z=0 plane.
   * @return The position as a 3-D vector.
   */
  Vector3d getPosition() const override {
    return position;
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
   */
  void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) {

    double x_min = map.min()[0];
    double x_max = map.max()[0];
    double y_min = map.min()[1];
    double y_max = map.max()[1];
    double z_min = map.min()[2];
    double z_max = map.max()[2];

    auto in_bounds = [&](const Vector3d &state) {
      std::vector<int> cells = map.meters2cells({state[0], state[1], state[2]});
      char collision = cmap[map.subv2ind_colmajor({cells[0], cells[1], cells[2]})];
      return ((state[0] >= x_min && state[0] <= x_max) &&
          (state[1] >= y_min && state[1] <= y_max) &&
          (state[2] >= z_min && state[2] <= z_max) && collision == '0');
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

    for (int t = 0; t < T; t++) {      // Update the state T times.
      position = position + normal_dist(W);
    }

    // Bound the target state inside the valid map region.
    position[0] = clamp(position[0], x_min, x_max);
    position[1] = clamp(position[1], y_min, y_max);
    position[2] = clamp(position[2], z_min, z_max);
  }
}; // End Class
} // End Namespace.
#endif //INFORMATION_GATHERING_LIBRARY_STATIC_TARGET_H
