//
// Created by brent on 4/7/17.
//

#include <igl/robot.h>
#include <igl/utils/utils_nx.h>
#include <iostream>
#include <set>
#include <igl/se3_pose.h>


template<class state>
void nx::Robot<state>::applyControl(std::vector<int> &action_idx, const int n_controls) {
  // If the robot has no actions, give a warning.
  if (action_idx.size() == 0) {
    std::cout << "WARNING: No control inputs available for robot to apply.\n";
  }

  std::vector<state> next_micro;
  int i = 0;
  for (auto it = action_idx.rbegin(); it != action_idx.rend() && i < n_controls; ++it, i++) {
    this->env->forward_action(this->getState(), action_idx.back(), next_micro);
    this->x = next_micro.back(); // Move the robot's position.
    action_idx.pop_back();
  }
}

template<class state>
state nx::Robot<state>::getState() const {
  return x;
}

template<class state>
void nx::Robot<state>::setState(state s) {
  x = s;
}

// Explicit Instantiations

template
class nx::Robot<std::vector<int>>;

template
class nx::Robot<nx::SE3Pose>;