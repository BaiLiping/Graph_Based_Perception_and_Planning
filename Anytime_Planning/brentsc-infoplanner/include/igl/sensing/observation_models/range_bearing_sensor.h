//
// Created by brent on 8/21/18.
//

#ifndef INFO_GATHERING_RANGEBEARINGMODEL_H
#define INFO_GATHERING_RANGEBEARINGMODEL_H

#include <random>
#include <Eigen/Dense>
#include <map>
#include <igl/sensing/sensor.h>
#include <igl/robot.h>
#include <igl/se3_pose.h>
#include <igl/sensing/observation_models/range_sensor.h>
#include <igl/sensing/observation_models/bearing_sensor.h>

using namespace Eigen;

namespace nx {

/**
 * The RangeBearingSensor class can simulate range and bearing measurements from the environment, and compute Jacobian
 * matrices of the corresponding sensor model for usage in filtering.
 */
class RangeBearingSensor : public nx::Sensor<SE3Pose> {

 public:
  // Range and bearing parameters.
  const double r_sense;
  const double fov;
  const double b_sigma;
  const double r_sigma;

  // Internal Range and Bearing Sensors
  RangeSensor range_sensor;
  BearingSensor bearing_sensor;

  // Map and CMap for ray-tracing.
  std::shared_ptr<nx::map_nd> map;
  const std::vector<char> cmap;

  /**
   * Constructs a RangeBearingSensor.
   * @param r_sense_ The sensing  radius.
   * @param fov_ The field of view.
   * @param b_sigma_ The standard deviation in bearing noise.
   * @param r_sigma_ The standard deviation in range noise.
   */
  RangeBearingSensor(double r_sense_,
                     double fov_,
                     double r_sigma_,
                     double b_sigma_,
                     std::shared_ptr<nx::map_nd> map,
                     const std::vector<char> &cmap)
      : Sensor(2), r_sense(r_sense_), fov(fov_), b_sigma(b_sigma_), r_sigma(r_sigma_), map(map), cmap(cmap),
        range_sensor(RangeSensor(0, r_sense, -fov/2, fov/2, 0, 360, r_sigma, map, cmap)),
        bearing_sensor(BearingSensor(0, r_sense, -fov/2, fov/2, b_sigma, map, cmap)) {}

  /**
   * Computes the observation model h(x,y) as a function of a robot state x and a single target state y.
   * @param x The sensor state x.
   * @param y The target state y.
   * @return The evaluated observation model h(x,y).
   */
  VectorXd observationModel(const SE3Pose &x, const VectorXd &y) const override {
    Vector2d z;
    z << range_sensor.observationModel(x, y), bearing_sensor.observationModel(x, y);
    return z;
  }

  /**
   * Computes a single noisy measurement vector of the target from a robot state x.
   * @param x The state sensing from.
   * @param target The target being sensed.
   * @return The resulting measurement.
   */
  Measurement sense(const SE3Pose &x, const Target &target) override {
    Measurement m_r = range_sensor.sense(x, target);
    Measurement m_b = bearing_sensor.sense(x, target);
    Vector2d z;
    z << m_r.z, m_b.z; // z = (Range, Bearing)
    Measurement m(z, target.ID, m_r.valid);
    return m;
  }

  void getJacobian(MatrixXd &H, MatrixXd &V, const SE3Pose &x, const VectorXd &y, bool check_valid) const override {
    V.setZero(); // Reset Jacobian Matrices
    H.setZero();
    double yaw = x.getYaw();
    // Concatenate Range and Bearing Jacobians
    Vector3d target_position;
    target_position << y[0], y[1], 0;
    if (!check_valid || range_sensor.isValid(x.orientation, x.position, target_position)) {
      MatrixXd H_r(2, 2), H_b(2, 2), V_r(1, 1), V_b(1, 1);
      range_sensor.getJacobian(H_r, V_r, x, y, false); // Don't need to repeat the valid check.
      bearing_sensor.getJacobian(H_b, V_b, x, y, false);
      H(0, 0) = H_r(0, 0);
      H(0, 1) = H_r(0, 1);
      H(1, 0) = H_b(0, 0);
      H(1, 1) = H_b(0, 1);
      V(0, 0) = V_r(0, 0);
      V(1, 1) = V_b(0, 0);
    }
    else {
      V(0, 0) = r_sigma * r_sigma;
      V(1, 1) = b_sigma * b_sigma;
    }
  }

  /**
   * Compute innovation between the measurement and observation models accounting for angle wrap-around.
   * @param measurement The measurement generated.
   * @param predicted_measurement The measurement predicted by an observation model.
   * @return The innovation between the two vectors.
   */
  VectorXd computeInnovation(const VectorXd &measurement,
                                     const VectorXd &predicted_measurement) const override {
    VectorXd innovation = measurement - predicted_measurement;
    for (int i = 0; i < innovation.rows(); i++)
      if (i % 2 == 1)
        innovation[i] = nx::restrict_angle(innovation[i]);
    return innovation;
  };
};

} // End namespace

#endif //INFO_GATHERING_RANGEBEARINGMODEL_H
