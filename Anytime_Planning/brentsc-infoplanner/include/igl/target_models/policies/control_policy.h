//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_TARGET_POLICY_H
#define INFORMATION_GATHERING_LIBRARY_TARGET_POLICY_H

#include <Eigen/Dense>
#include <functional>

using namespace Eigen;
namespace nx {

/**
 * @brief ControlPolicy class is used to define a control policy for targets in an environment.
 * @tparam N The dimension of the state.
 * @tparam M The dimension of the control input.
 */
template<int N, int M>
class ControlPolicy {
 public:

  // The Feedback Control Policy
  std::function<VectorXd(const VectorXd &)> compute_control;

  /**
   * @brief Construct a ControlPolicy with specified function.
   * @param function The policy.
   */
  ControlPolicy(const std::function<VectorXd(const VectorXd &)> &function)
      : compute_control(function) {}
  /**
   * @brief Uses the underlying control policy to generate a feedback control input.
   * @param state The current state
   * @return The computed control value.
   */
  VectorXd computeControl(const VectorXd &state) {
    return compute_control(state);
  }
}; // End class
} // End namespace


#endif //INFORMATION_GATHERING_LIBRARY_TARGET_POLICY_H
