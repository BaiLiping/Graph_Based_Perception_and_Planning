//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_BELIEF_H
#define INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_BELIEF_H

#include <igl/target_models/dynamics/double_integrator_2d.h>
#include <igl/target_models/info_target_model.h>

namespace nx {

/**
 * @brief The DoubleIntegrator2DBelief object manages a Gaussian Belief over a double integrating-2D target.
 */
class DoubleIntegrator2DBelief : public DoubleIntegrator2D, public InfoTarget {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DoubleIntegrator2DBelief(const DoubleIntegrator2D &target, const MatrixXd &cov) :
      DoubleIntegrator2D(target), InfoTarget(target, cov) {}

  void updateBelief(const VectorXd &mean, const MatrixXd &cov, const nx::map_nd &map,
                    const std::vector<char> &cmap) override {
    DoubleIntegrator2D::position = mean.head(2);
    DoubleIntegrator2D::velocity = mean.tail(2);
    covariance = cov;

    // Restrict the belief to the map boundaries.
    DoubleIntegrator2D::restrict_position(map, cmap);
  }

  virtual Vector3d getPosition() const override {
    return DoubleIntegrator2D::getPosition();
  }

  virtual VectorXd getState() const override {
    return DoubleIntegrator2D::getState();
  }

  virtual MatrixXd getJacobian() const override {
    return DoubleIntegrator2D::getJacobian();
  }

  virtual MatrixXd getNoise() const override {
    return DoubleIntegrator2D::getNoise();
  }

  virtual void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) override {
    DoubleIntegrator2D::forwardSimulate(T, map, cmap);
  }

}; // End class
} // End namespace

#endif //INFORMATION_GATHERING_LIBRARY_DOUBLE_INTEGRATOR_2D_BELIEF_H
