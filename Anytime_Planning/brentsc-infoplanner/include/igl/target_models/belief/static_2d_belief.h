//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_STATIC_2D_BELIEF_H
#define INFORMATION_GATHERING_LIBRARY_STATIC_2D_BELIEF_H

#include <igl/target_models/dynamics/static_2d.h>
#include <igl/target_models/info_target_model.h>

namespace nx {

/**
 * @brief The Static2DBelief object manages a Gaussian Belief over a static 2D target.
 */
class Static2DBelief : public Static2D, public InfoTarget {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Static2DBelief(const Static2D &target, const MatrixXd &cov) :
      Static2D(target), InfoTarget(target, cov) {}

  void updateBelief(const VectorXd &mean, const MatrixXd &cov, const nx::map_nd &map,
                    const std::vector<char> &cmap) override {
    // Update Mean
    Static2D::position = mean.head(2);
    covariance = cov;

    // Restrict the belief to the map boundaries.
    Static2D::restrict_position(map, cmap);
  }

  virtual Vector3d getPosition() const override {
    return Static2D::getPosition();
  }

  virtual VectorXd getState() const override {
    return Static2D::getState();
  }

  virtual MatrixXd getJacobian() const override {
    return Static2D::getJacobian();
  }

  virtual MatrixXd getNoise() const override {
    return Static2D::getNoise();
  }

  virtual void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) override {
    Static2D::forwardSimulate(T, map, cmap);
  }
}; // End class
} // End namespace

#endif //INFORMATION_GATHERING_LIBRARY_STATIC_2D_BELIEF_H
