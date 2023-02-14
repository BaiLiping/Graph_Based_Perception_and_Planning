//
// Created by brent on 11/29/18.
//

#ifndef INFORMATION_GATHERING_LIBRARY_MULTI_TARGET_FILTER_H
#define INFORMATION_GATHERING_LIBRARY_MULTI_TARGET_FILTER_H

#include <igl/target_models/info_target_model.h>
#include <igl/robot.h>
#include <igl/estimation/kalman_filter.h>
#include <iostream>

namespace nx {

/**
 * The class MultiTargetFilter manages the state estimation of a set of multiple targets from a TargetModel,
 * as estimated from a robot.
 */
template<class state>
class MultiTargetFilter {
 public:

  /**
   * Computes the Kalman Filter covariance matrix for the robot given, at state x_t, a target prediction of y_t, and
   * a covariance prior of cov_prior.
   * @param robot The robot using the filter.
   * @param x_t The state to evaluate Jacobians with respect to.
   * @param y_t The target to evaluate Jacobians with respect to.
   * @param cov_prior The prior covariance matrix.
   * @return The posterior covariance matrix.
   */
  static MatrixXd MultiTargetKFCovariance(const Robot<state> &robot,
                                          const state &x_t,
                                          const VectorXd &y_t,
                                          const MatrixXd &cov_prior) {
    // Compute dimensions of the problem.
    int num_targets_known = robot.tmm->num_targets();
    int y_dim = robot.tmm->target_dim / num_targets_known;
    int z_dim = robot.sensor->z_dim;
    // Allocate matrices.
    MatrixXd A(num_targets_known * y_dim, num_targets_known * y_dim);
    MatrixXd W(num_targets_known * y_dim, num_targets_known * y_dim);
    MatrixXd H(num_targets_known * z_dim, num_targets_known * y_dim);
    MatrixXd V(num_targets_known * z_dim, num_targets_known * z_dim);
    // Get system and observation matrices from robot's properties.
    robot.tmm->getJacobian(A, W);
    robot.sensor->getJacobian(H, V, x_t, robot.tmm, y_t);

    return KalmanFilter::KFCovariance(cov_prior, A, W, H, V);
  }

  /**
   * Computes a full Kalman Filter update of a TargetModel corresponding to the robot.
   * @param measurements The vector of Measurements from all the targets.
   * @param robot The robot who is filtering.
   * @param debug A flag to print out debugging messages.
   * @return The posterior GaussianBelief of the target.
   */
  static GaussianBelief MultiTargetKF(const std::vector<Measurement> &measurements,
                                      const Robot<state> &robot,
                                      bool debug = false) {
    state x_t = robot.getState();
    VectorXd mean_prior = robot.tmm->getTargetState();
    MatrixXd cov_prior = robot.tmm->getCovarianceMatrix();
    // Get problem dimension.
    int num_targets = robot.tmm->num_targets();
    int y_dim = robot.tmm->target_dim / num_targets;
    int z_dim = robot.sensor->z_dim;
    // Allocate matrices.
    MatrixXd A(num_targets * y_dim, num_targets * y_dim);
    MatrixXd W(num_targets * y_dim, num_targets * y_dim);
    MatrixXd H(num_targets * z_dim, num_targets * y_dim); // Jacobian w.r.t. range bearing for y_dim
    MatrixXd V(num_targets * z_dim, num_targets * z_dim); // target

    // Get Target Motion Model, and Sensor Observation Model
    std::vector<bool> validity(num_targets);
    std::transform(measurements.begin(), measurements.end(), validity.begin(), [&](Measurement m) { return m.valid; });
    robot.tmm->getJacobian(A, W);

    VectorXd innovation(z_dim * num_targets);
    H.setZero();
    V.setZero();
    for (int i = 0; i < measurements.size(); i++) {
      auto target = robot.tmm->getTargetByID(measurements[i].ID);
      VectorXd y_predict = target->predictState(1);
      // Compute Jacobian entries only for valid measurements
      MatrixXd H_i(z_dim, y_dim);
      MatrixXd V_i(z_dim, z_dim);
      robot.sensor->getJacobian(H_i, V_i, x_t, y_predict, false); // Don't check validity again.
      if (validity[i])
        H.block(i * z_dim, i * y_dim, z_dim, y_dim) = H_i;
      V.block(i * z_dim, i * z_dim, z_dim, z_dim) = V_i;

      // Compute Innovation
      VectorXd z = measurements[i].z;
      VectorXd h_xy = robot.sensor->observationModel(x_t, y_predict);
      innovation.segment(i * z_dim, z_dim) = robot.sensor->computeInnovation(z, h_xy);
      if (debug) {
        std::cout << "Target: id=" << target->ID << " state=" << target->getPosition() << std::endl;
        std::cout << "Measurement: " << z << std::endl;
        std::cout << "H_XY: " << h_xy << std::endl;
      }
    }
    GaussianBelief result = KalmanFilter::KF(mean_prior, cov_prior, A, W, H, V, innovation, debug);
    return result;
  }

};
}; // End namespace

#endif //INFORMATION_GATHERING_LIBRARY_MULTI_TARGET_FILTER_H
