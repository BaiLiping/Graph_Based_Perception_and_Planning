//
// Created by brent on 8/21/18.
//

#ifndef INFO_GATHERING_INFOTARGETMODEL_H
#define INFO_GATHERING_INFOTARGETMODEL_H

#include <igl/target_models/target_model.h>
#include <igl/estimation/kalman_filter.h>

namespace nx {

/**
 * The InfoTarget class extends the Target class to include a Covariance matrix over the Target state.
 */
class InfoTarget : public Target {
 protected:
  MatrixXd covariance; // Covariance over the Target.

 public:
  /**
   * @brief Constructs the InfoTarget
   * @param target
   * @param cov
   */
  InfoTarget(const Target &target, const MatrixXd &cov) : Target(target),
                                                          covariance(cov) {}
  /**
   * @brief Predicts the target State evolution for T timesteps.
   * @param T The number of timesteps to predict over.
   * @return The resulting state prediction.
   */
  virtual VectorXd predictState(int T) const {
    return getState();
  }

  /**
   * @brief Returns the mean of the current target distribution.
   * @return The current target mean.
   */
  VectorXd getState() const {
    return getState();
  }

  /**
   * @brief Returns the covariance of the target belief.
   * @return The covariance of the target belief.
   */
  virtual MatrixXd getCovariance() const {
    return covariance;
  }

  /**
   * @brief Updates the Gaussian distribution over the state.
   * @param mean The updated mean of the distribution.
   * @param cov The updated covariance of the distribution.
   * @param map The map for boundary checking.
   * @param cmap The costmap for collision checking.
   */
  virtual void updateBelief(const VectorXd &mean, const MatrixXd &cov,
                            const nx::map_nd &map, const std::vector<char> &cmap) = 0;
}; // End class

/**
 * The infoTargetModel class extends targetModel in order to maintain a Gaussian distribution over the
 * target state (mean, and covariance). This is used for a robot's internal representation of the target.
 */
class InfoTargetModel : public TargetModel {
 public:

  /**
   * Construct an empty InfoTargetModel with a map of the environment.
   */
  InfoTargetModel(const nx::map_nd &map, const std::vector<char> &cmap) : TargetModel(map, cmap) {};

  bool addTarget(int ID, std::shared_ptr<InfoTarget> info_target) {
    bool result = AddSharedTarget(ID, info_target);
    return result;
  }

  /**
   * Returns the joint system matrix A of the target model.
   * @return The system matrix.
   */
  MatrixXd getCovarianceMatrix() const {
    MatrixXd result = MatrixXd::Zero(target_dim, target_dim);
    // Loop over all targets to construct the joint system matrix.
    int index = 0;
    for (const auto &pair: targets) {
      std::shared_ptr<InfoTarget> target = std::static_pointer_cast<InfoTarget>(pair.second);
      result.block(index, index, target->y_dim, target->y_dim) = target->getCovariance();
      index += target->y_dim;
    }
    // Return the result.
    return result;
  }

  /**
   * Updates the Information State of the Target Model.
   * @param y_ The new mean.
   * @param Sigma_ The new covariance.
   */
  void updateBelief(const VectorXd &mean, const MatrixXd &covariance) {
    int index = 0;
    for (auto &pair : targets) {
      std::shared_ptr<InfoTarget> target = std::static_pointer_cast<InfoTarget>(pair.second);
      // Get the correct mean and covariance
      target->updateBelief(mean.segment(index, target->y_dim),
                           covariance.block(index, index, target->y_dim, target->y_dim), *map, cmap);
      index += target->y_dim;
    }
  }

  /**
   * Assigns the Jacobian for the Target motion model.
   * @param A_
   * @param W_
   */
  void getJacobian(MatrixXd &A_, MatrixXd &W_) const {
    A_ = getSystemMatrix();
    W_ = getNoiseMatrix();
  }

  /**
   * Predicts a target trajectory of horizon T, and returns the result. The first entry is the current state.
   * @param T The time horizon to predict the trajectory over.
   * @return The vector of predicted target states over the horizon.
   */
  std::vector<VectorXd> predictTargetState(int T) const {
    std::vector<VectorXd> result;
    VectorXd target_state = getTargetState();
    MatrixXd A = getSystemMatrix();
    for (int t = 0; t <= T; t++) {
      result.push_back(target_state);
      target_state = A * target_state;
    }
    return result;
  };

  /**
   * Returns the InfoTarget by ID
   * @param ID The ID of the target to return.
   * @return The InfoTarget queried for.
   */
  std::shared_ptr<InfoTarget> getTargetByID(int ID) const {
    return std::static_pointer_cast<InfoTarget>(targets.at(ID));
  }

}; // End class
} // end namespace
#endif //INFO_GATHERING_INFOTARGETMODEL_H
