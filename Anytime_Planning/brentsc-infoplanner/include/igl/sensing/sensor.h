//
// Created by brent on 8/21/18.
//

#ifndef INFO_GATHERING_SENSOR_H
#define INFO_GATHERING_SENSOR_H

#include <random>
#include <iostream>
#include <Eigen/Dense>
#include <map>
#include <igl/target_models/info_target_model.h>

using namespace Eigen;

namespace nx {

/**
 * The Measurement structure contains the measurement information of a target, as detected by a sensor.
 */
struct Measurement {
  VectorXd z; // Noisy sensor measurement.
  int ID; // ID
  int valid; // Valid Flag
  unsigned z_dim; // Dimension of the measurement per target.

  /**
   * Constructs a measurement.
   * @param z The measurement vector z.
   * @param z_with_noise The measurement with Additive noise.
   * @param ID The ID
   * @param valid The validity of the measurement.
   */
  Measurement(const VectorXd &z, int ID, int valid) :
      z(z), ID(ID), valid(valid), z_dim(z.rows()) {};
};

/**
 * Generic interface for an abstract sensor type. Supports generating single measurements of a target from a state, a
 * vector of measurements of multiple targets from a TargetModel, and computation of Jacobian matrices.
 */
template<class state>
class Sensor {
 public:
  int z_dim; // Dimension to size matrices correctly.

  Sensor(int z_dim) : z_dim(z_dim) {};

  /**
   * Computes the observation model h(x,y) as a function of a robot state x and a single target state y.
   * @param x The sensor state x.
   * @param y The target state y.
   * @return The evaluated observation model h(x,y).
   */
  virtual VectorXd observationModel(const state &x, const VectorXd &y) const = 0;

  /**
   * Computes a single noisy measurement vector of the target from a robot state x.
   * @param x The state sensing from.
   * @param y The target being sensed.
   * @return The resulting measurement.
   */
  virtual Measurement sense(const state &x, const Target &y) = 0;

  /**
   * Compute innovation between the measurement and observation models (accounting for any non-linearities).
   * @param measurement The measurement generated.
   * @param predicted_measurement The measurement predicted by an observation model.
   * @return The innovation between the two vectors.
   */
  virtual VectorXd computeInnovation(const VectorXd &measurement, const VectorXd &predicted_measurement) const {
    return measurement - predicted_measurement;
  };

  /**
   * Computes the Jacobian matrix (linearization of the observation model) for a single target y.
   * @param H The linearization of the observation model.
   * @param V The noise matrix.
   * @param x The state sensing from.
   * @param y The target being sensed.
   */
  virtual void getJacobian(MatrixXd &H, MatrixXd &V, const state &x, const VectorXd &y, bool check_valid=true) const = 0;

  /**
   * @brief Returns the maximum bounding Sensor Information Matrix H(x)'inv(V)H(x), attainable by the state x for a
   * given target model tmm.
   * @param x The current state.
   * @param y The predicted target state.
   * @param tmm The current target model.
   * @param T The number of timesteps to compute the reachable space.
   * @return The maximum bounding Sensor Information Matrix.
   */
  virtual MatrixXd maxSensorMatrix(const state &x, const VectorXd &y, const std::shared_ptr<TargetModel> &tmm, int T) const {
      return 1e10* MatrixXd::Identity(y.rows(), y.rows());
  }

  /*****************************************************************************************
   * Common functions for multiple targets.
   *****************************************************************************************/

  /**
   * Returns a vector of sensor measurements from the environment.
   * @param x The state to generate measurements from.
   * @param tmm The Target Model.
   * @return The vector of sensor measurements generated.
   */
  virtual std::vector<Measurement> senseMultiple(const state &x, const std::shared_ptr<TargetModel> &tmm) {
    std::vector<Measurement> output;
    for (const auto &pair : tmm->targets) {
      output.push_back(sense(x, *pair.second));
    }
    return output;     // Return Vector of measurements.
  }

  /**
   * Computes the Jacobian of the sensor measurement model.
   * @param H The Jacobian of the measurement model.
   * @param V The covariance matrix.
   * @param x The robot state to compute the Jacobian with respect to.
   * @param y The target state to compute the Jacobian with respect to.
   * @param measurements Vector of measurements used for checking if
   */
  virtual void getJacobian(MatrixXd &H,
                           MatrixXd &V,
                           const state &x,
                           const std::shared_ptr<TargetModel> &tmm,
                           const VectorXd &y) {
    V.setZero();     // Reset Block Matrices
    H.setZero();
    // Iterate over all targets.
    int index = 0;
    for (const auto &pair : tmm->targets) {
      auto target = pair.second;
      int y_dim = target->y_dim;
      VectorXd y_predict = y.segment(index * y_dim, y_dim);  // Predict next target state
      MatrixXd H_i(z_dim, y_dim);
      MatrixXd V_i(z_dim, z_dim);
      getJacobian(H_i, V_i, x, y_predict);
      H.block(index * z_dim, index * y_dim, z_dim, y_dim) = H_i;
      V.block(index * z_dim, index * z_dim, z_dim, z_dim) = V_i;
      index++;
    } // End Loop
  }
};

}

#endif //INFO_GATHERING_SENSOR_H
