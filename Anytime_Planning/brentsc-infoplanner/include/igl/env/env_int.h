#ifndef __ENV_INT_H_
#define __ENV_INT_H_

#include <vector>                           // std::vector
#include <Eigen/Dense>
#include <unordered_map>                  // std::unordered_map
#include <map>
#include <iostream>
#include <memory>   // std::unique_ptr
#include <igl/mapping/map_nx.h>
#include <igl/control/motion_primitive.h>

using namespace Eigen;

namespace nx {

template<class state>
class Environment {

 public:
  std::shared_ptr<map_nd> map; // Map parameters
  std::shared_ptr<std::vector<char>> cmap_; // Collision Map
  state goal_pose; // Goal pose. Optional
  typedef MotionPrimitive<std::array<double, 3>, std::array<double, 2>> MPrim;
  std::vector<MPrim> mprim_; // TODO Redesign this. MPrim are public for now..

  /**
   * Constructs an environment interface.
   * @param MAP_ptr_in The unique_ptr to the map information.
   * @param cmap The unique_ptr to the costMap.
   * @param goal The optional goal coordinate.
   */
  Environment(const nx::map_nd &map_in,
              const std::vector<char> &cmap,
              state goal = state())
      : cmap_(std::make_shared<std::vector<char>>(cmap)),
        map(std::make_shared<nx::map_nd>(map_in)), goal_pose(goal) {}

  /**
   * Computes the successor nodes from state curr, taking into account the costMap.
   * @param curr The current state.
   * @param succ The list of successors to be computed.
   * @param succ_idx The linear indices of the successors.
   * @param succ_cost
   * @param action_idx
   */
  virtual void get_succ(const state &curr,
                        std::vector<state> &succ,
                        std::vector<double> &succ_cost,
                        std::vector<int> &action_idx) const = 0;

  /**
   * Computes the list of micro-states generated from applying the action with action_id.
   * @param curr Current state.
   * @param action_id Action index to apply.
   * @param next_micro List of micro-states generated along the action.
   */
  virtual void forward_action(const state &curr, int action_id, std::vector<state> &next_micro) const = 0;

  /**
   * Converts a state to its corresponding linear index in the map.
   * @return The linear index of the state.
   */
  virtual int state_to_idx(const state &) const = 0;

  /**
   * Converts a state to its corresponding cell coordinates in the map.
   * @return The cell coordinates of the state.
   */
  virtual std::vector<int> state_to_cell(const state &) const = 0;

  /**
   * Returns the pose of the state, projecected into the SE(2) space, i.e. 2.5D position (X,Y,Yaw).
   * @return The SE(2) Pose of the state.
   */
  virtual Vector3d state_to_SE2(const state &) const = 0;

  /**
   * Computes the distance metric between two states.
   * @return The distnace between the two states.
   */
  virtual double ComputeStateMetric(const state &, const state &) const = 0;

  virtual std::vector<char> GetCostMap() {
    std::vector<char> cmap = *cmap_;

    return std::vector<char>(cmap.begin(), cmap.end());
  }
};
} // End namespace


#endif
