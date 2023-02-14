//
// Created by brent on 4/7/17.
//

#ifndef INFO_GATHERING_ROBOT_H
#define INFO_GATHERING_ROBOT_H

#include <time.h>       /* time */
#include <stdlib.h>
#include <igl/sensing/sensor.h>
#include <igl/target_models/info_target_model.h>
#include <igl/env/env_int.h>

namespace nx {

/**
 * The Robot class is an abstraction of a robot consisting of a state, dynamics, and target and observation models
 * related to an information gathering task it will aim to achieve.
 */
template<class state>
class Robot {

 public:
  std::shared_ptr<Environment<state>> env; // The environment may be shared.
  std::shared_ptr<InfoTargetModel> tmm; // Each robot should have its own target model.
  std::shared_ptr<Sensor<state>> sensor; // Sensors may be shared.

  /**
   * Constructs a robot with given initial state, environment, target model, and sensor.
   * @param x0 The initial robot state.
   * @param env The robot environment.
   * @param tmm The target motion model.
   * @param sensor The robot sensor.
   */
  Robot(state x0, std::shared_ptr<Environment<state>> env, std::shared_ptr<InfoTargetModel> tmm,
        std::shared_ptr<Sensor<state>> sensor)
      : x(x0), env(env), tmm(std::move(tmm)), sensor(std::move(sensor)) {}

  /**
   * Applies a sequence of controls to the robot's state, modifying its current state.
   * @tparam state The state space of the robot.
   * @param action_idx The vector of indices in the action space to apply.
   * @param n_controls The number of control inputs in the sequence to apply.
   */
  void applyControl(std::vector<int> &action_idx, const int n_control=1);

  /**
   * Returns the state of the robot in its templated type.
   * @tparam state The state space of the robot.
   * @return The templated state of the robot.
   */
  state getState() const;

  /**
   * Assigns the current robot state to s.
   * @param s The new state to be assigned.
   */
  void setState(state s);

 protected:
  state x; // State of the robot
};
}
#endif //INFO_GATHERING_ROBOT_H
