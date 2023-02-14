//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_DETERMINANT_H
#define INFORMATION_GATHERING_LIBRARY_DETERMINANT_H
#include <igl/planning/cost_functions/cost_function.h>

class DeterminantCost : public CostFunction {
  /**
   * @brief Computes a scalar cost function given a Covariance matrix and timestep.
   * @param Sigma The covariance matrix.
   * @param t The timestep.
   * @return The computed cost.
   */
  double computeCost(const MatrixXd &Sigma, int t = 0) const override {
    return Sigma.determinant();
  }
};
#endif //INFORMATION_GATHERING_LIBRARY_DETERMINANT_H
