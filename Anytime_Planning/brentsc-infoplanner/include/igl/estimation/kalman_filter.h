//
// Created by brent on 8/21/18.
//

#ifndef INFO_GATHERING_KF_H
#define INFO_GATHERING_KF_H

#include <Eigen/Dense>
#include <memory>
#include <iostream>

using namespace Eigen;

namespace nx {

/**
 * The GaussianBelief struct stores a mean and covariance of a multivariate Gaussian distribution.
 */
struct GaussianBelief {

  VectorXd mean; // The mean of the Gaussian.
  MatrixXd cov; // The covariance matrix of the Gaussian.

  /**
   * Constructs the Gaussian Belief from a mean and covariance.
   * @param mean The mean of the Gaussian.
   * @param covariance The covariance of the Gaussian.
   */
  GaussianBelief(const VectorXd &mean, const MatrixXd &covariance) : mean(mean), cov(covariance) {};

};

/**
 * The KalmanFilter class contains static methods which execute the predict and update steps of a Kalman Filter, both
 * with and without measurements.
 */
class KalmanFilter {
 public:
  /**
   * The KFCovariance function takes as input a prior covariance matrix, and computes the Kalman filter prediction and
   * update steps to calculate the posterior covariance matrix. Note that the covariance posterior is deterministic
   * and does not depend on the measurement values.
   * @param cov_prior The prior covariance matrix.
   * @param A The system dynamics matrix.
   * @param W The system noise matrix.
   * @param H The observation model matrix.
   * @param V The observation model noise matrix.
   * @return The posterior covariance matrix.
   */
  static MatrixXd KFCovariance(const MatrixXd &cov_prior,
                               const MatrixXd &A, const MatrixXd &W, const MatrixXd &H, const MatrixXd &V) {

    // Predict
    MatrixXd cov_predict = A * cov_prior * A.transpose() + W;

    // Update
    MatrixXd R = H * cov_predict * H.transpose() + V;
    MatrixXd K = cov_predict * H.transpose() * R.inverse(); // Kalman Gain
    MatrixXd C = MatrixXd::Identity(cov_predict.rows(), cov_predict.rows()) - K * H;
    MatrixXd cov_update = C * cov_predict;

    // Return Result
    return cov_update;
  }

  /**
   * The KF function takes as input the innovation vector, and prior Gaussian distribution to compute the Kalman Filter
   * prediction and update steps to compute a posterior Gaussian distribution.
   * @param innovation The innovation vector, i.e. z - h(mean).
   * @param mean_prior The prior mean.
   * @param cov_prior The prior covariance matrix.
   * @param A The system dynamics matrix.
   * @param W The system noise matrix.
   * @param H The observation model matrix.
   * @param V The observation model noise matrix.
   * @return The posterior Gaussian belief.
   */
  static GaussianBelief KF(const VectorXd &mean_prior,
                           const MatrixXd &cov_prior,
                           const MatrixXd &A,
                           const MatrixXd &W,
                           const MatrixXd &H,
                           const MatrixXd &V,
                           const VectorXd &innovation,
                           int debug = 0) {
    // Predict
    VectorXd mean_predict = A * mean_prior;
    MatrixXd cov_predict = A * cov_prior * A.transpose() + W;

    // Update
    MatrixXd R = H * cov_predict * H.transpose() + V;
    MatrixXd K = cov_predict * H.transpose() * R.inverse(); // Kalman Gain
    MatrixXd C = MatrixXd::Identity(cov_predict.rows(), cov_predict.rows()) - K * H;
    MatrixXd cov_update = C * cov_predict;
    VectorXd mean_update = mean_predict + K * innovation;

    if (debug)
      PrintDebug(innovation, mean_prior, cov_prior, mean_predict, cov_predict, A, W, H, V, K, R, mean_update, cov_update);
    // Return Result
    GaussianBelief output(mean_update, cov_update);
    return output;
  }

 private:

  /**
   * Prints Kalman Filter matrices for debugging purposes.
   */
  static void PrintDebug(const VectorXd &innovation,
                         const VectorXd &mean_prior,
                         const MatrixXd &cov_prior,
                         const VectorXd &mean_predict,
                         const MatrixXd &cov_predict,
                         const MatrixXd &A,
                         const MatrixXd &W,
                         const MatrixXd &H,
                         const MatrixXd &V,
                         const MatrixXd &K,
                         const MatrixXd &R,
                         const VectorXd &mean_update,
                         const MatrixXd &cov_update) {
    // Debug
    std::cout << "Kalman Gain: \n" << K << std::endl;
    std::cout << "R Mat: \n" << R << std::endl;
    std::cout << "H Mat: \n" << H << std::endl;
    std::cout << "V Mat: \n" << V << std::endl;
    std::cout << "A Mat: \n" << A << std::endl;
    std::cout << "W Mat: \n" << W << std::endl;
    std::cout << "Mean prior: \n" << mean_prior << std::endl;
    std::cout << "Sigma prior: \n" << cov_prior << std::endl;
    std::cout << "Mean pred: \n" << mean_predict << std::endl;
    std::cout << "Sigma pred: \n" << cov_predict << std::endl;
    std::cout << "Innovation (z-h_xy) = \n" << innovation << std::endl;
    std::cout << "Mean update: \n" << mean_update << std::endl;
    std::cout << "Sigma update: \n" << cov_update << std::endl;

  }
}; // End class

} // End namespace


#endif //INFO_GATHERING_KF_H

