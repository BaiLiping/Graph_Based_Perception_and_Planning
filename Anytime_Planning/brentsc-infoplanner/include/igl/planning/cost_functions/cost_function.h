//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_COST_FUNCTION_H
#define INFORMATION_GATHERING_LIBRARY_COST_FUNCTION_H
#include <Eigen/Dense>

using namespace Eigen;

/**
 * @brief The CostFunction class stores object references capable of computing stage cost functions on
 * given an input Covariance Matrix and timestep.
 */
class CostFunction {
 public:
  /**
   * @brief Computes a scalar cost function given a Covariance matrix and timestep.
   * @param Sigma The covariance matrix.
   * @param t The timestep. Defaults argument is zero if it is un-used.
   * @return The computed cost.
   */
  virtual double computeCost(const MatrixXd &Sigma, int t = 0) const {
    return std::log(Sigma.determinant());
  }
};

#endif //INFORMATION_GATHERING_LIBRARY_COST_FUNCTION_H
