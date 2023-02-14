//
// Created by brent on 4/24/17.
//

#ifndef INFO_GATHERING_INFO_PLANNER_H
#define INFO_GATHERING_INFO_PLANNER_H

#include <boost/heap/d_ary_heap.hpp>      // boost::heap::d_ary_heap
#include <memory>                         // std::shared_ptr
#include <limits>                         // std::numeric_limits
#include <vector>                         // std::vector
#include <unordered_map>                  // std::unordered_map
#include <array>                          // std::array
#include <list>                           // std::list
#include <chrono>                         // std::chrono::high_resolution_clock
#include <Eigen/Dense>
#include <map>
#include <iostream>
#include <igl/planning/cost_functions/cost_function.h>

namespace nx {
using namespace Eigen;

// Forward declaration
template<class state>
class Environment;

template<class state>
class Robot;

// heap element comparison
template<class infostate>
struct compare_pair {
  bool operator()(const std::pair<double, std::shared_ptr<infostate>> &p1,
                  const std::pair<double, std::shared_ptr<infostate>> &p2) const {
    if ((p1.first >= p2.first - 0.000001) && (p1.first <= p2.first + 0.000001)) {
      // if equal compare gvals
      return (p1.second->g) < (p2.second->g);
    }
    return p1.first > p2.first;
  }
};

template<class infostate>
using hashMap = std::unordered_map<int, std::shared_ptr<infostate> >;

template<class infostate>
using priorityQueue = boost::heap::d_ary_heap<std::pair<double, std::shared_ptr<infostate>>,
                                              boost::heap::mutable_<true>,
                                              boost::heap::arity<2>,
                                              boost::heap::compare<compare_pair<infostate> >>;
template<class infostate>
using covQueue = boost::heap::d_ary_heap<std::pair<double, std::shared_ptr<infostate>>,
                                         boost::heap::mutable_<true>,
                                         boost::heap::arity<2>,
                                         boost::heap::compare<compare_pair<infostate> >>;

template<class state>
struct InfoState {
  state coord; // Spatial location of the InfoState.
  MatrixXd Sigma; // Information state of the InfoState.
  int t; // Timestep this Node is reached.

  std::shared_ptr<InfoState<state>> parent{nullptr}; // pointer to parent node
  int parent_action_id = -1; // Parent Action Index.

  // Each state maintains a list of other states at its own spatial location.
  std::shared_ptr<covQueue<InfoState<state>>> state_list;

  int hashkey;
  // pointer to heap location
  typename priorityQueue<InfoState<state>>::handle_type heapkey;

  bool is_open{false}; // Flag indicating if the Node is open or not.
  bool is_closed{false}; // Flag indicating if the Node is closed for further expansion.
  int iteration_opened {0};
  int iteration_closed {0};

  double g = std::numeric_limits<double>::infinity(); // Cost to Reach this state.
  double f = 0;
  double h = 0;

  InfoState(int hashkey, const state &coord, MatrixXd &Sigma, int t, bool new_state = false)
      : hashkey(hashkey), coord(coord), Sigma(Sigma), t(t)//, parent(nullptr)
  {
    if (new_state) {
      state_list.reset(new covQueue<InfoState<state>>());
    }
  }
};

template<class state>
struct D {
  /**
   * This destructor destroys the references to all other InfoStates at the same spatial location.
   * @param p
   */
  void operator()(InfoState<state> *p) const {
    // EXTREMELY IMPORTANT!!
    delete p;
  }

};

template<class state>
struct InfoStateSpace {
  priorityQueue<InfoState<state>> pq;
  hashMap<InfoState<state>> hm;

  InfoStateSpace() {}

  ~InfoStateSpace() {
    for (auto it = hm.begin(); it != hm.end(); it++) {
      std::shared_ptr<InfoState<state>> currNode_pt = it->second;
      for (auto jt = it->second->state_list->begin(); jt != it->second->state_list->end(); jt++) {
        std::shared_ptr<InfoState<state>> currNode_pt = jt->second;
      }
      currNode_pt->state_list->clear();
    }
  }
};

/**
 * Output type of the planner.
 * @tparam state The state space being planned in.
 */
template<class state>
struct PlannerOutput {
  std::list<state> path;
  std::list<state> observation_points;
  std::vector<int> action_idx;
  std::vector<VectorXd> target_path;
  MatrixXd outputCovariance;
  std::vector<std::list<state>> all_paths;
  // Cost values
  double cost;
  double cost_per_node;
  double nodes;
  double total_node_cost;


  std::map<int, std::list<state>> opened_list;
  std::map<int, state> closed_list;

};

template<class state>
class InfoPlanner {

 private:
  std::shared_ptr<CostFunction> cost_function;

 public:

  /**
   * @brief Default constructor for the planner. Defaults to the Log-Determinant cost.
   */
  InfoPlanner() : cost_function(std::make_shared<CostFunction>()) {}

  /**
   * @brief Initialize the planner with a specified Cost Function.
   * @param cost_function The pointer to the cost function object.
   */
  InfoPlanner(std::shared_ptr<CostFunction> cost_function) : cost_function(cost_function) {}

  /**
   * @brief This algorithm attempts to improve the quality of the search tree given by S0, by successively making calls to the
   * ImprovePath subroutine until time runs out. Each call to improvePath will decrease the epsilon and delta parameters.
   * @tparam robot The robot being planned for.
   * @param T The desired planning horizon to compute paths for.
   * @param del The initial value for the delta parameter.
   * @param epsilon The initial value of the epsilon parameter.
   * @param allocated_time_secs The allocated search time. The algorithm will return faster than this.
   * @param debug Flag to print debug messages
   * @param log Flag to log nodes expanded or not.
   */
  PlannerOutput<state> PlanARVI(const Robot<state> &robot,
                                const int &T,
                                const double &del = 5,
                                const double &epsilon = std::numeric_limits<double>::infinity(),
                                const double &allocated_time_secs = 0.1,
                                const int &debug = 0,
                                bool log = false) const;

  /**
    * @brief This runs the Reduced Value Iteration algorithm for active information acquisition, for a fixed epsilon and delta
    * given to the planner.
    * @tparam robot The robot being planned for.
    * @param T The desired planning horizon to compute paths for.
    * @param del The initial value for the delta parameter.
    * @param epsilon The initial value of the epsilon parameter.
    * @param debug Flag to print debug messages
    * @param log Flag to log nodes expanded or not.
    */
  PlannerOutput<state> PlanRVI(const Robot<state> &robot,
                               const int &T,
                               const double &del = 5,
                               const double &epsilon = std::numeric_limits<double>::infinity(),
                               const int &debug = 0,
                               bool log = false) const;

  /**
  * @brief This runs the Forward Value Iteration algorithm for active information acquisition. This is an exhaustive
  * search and should only be used for comparison purposes.
  * @tparam robot The robot being planned for.
  * @param T The desired planning horizon to compute paths for.
  * @param del The initial value for the delta parameter.
  * @param epsilon The initial value of the epsilon parameter.
  * @param debug Flag to print debug messages
  * @param log Flag to log nodes expanded or not.
  */
  PlannerOutput<state> PlanFVI(const Robot<state> &robot,
                               const int &T,
                               const int &debug = 0,
                               bool log = false) const;

  PlannerOutput<state> PlanAstar(const Robot<state> &robot,
                                 const int &T,
                                 const double &epsilon = 1,
                                 const int &debug = 0,
                                 bool log = false) const;

 private:

  /**
   * The improvePath subroutine used by the main algorithms. This will update a search tree S by adding new nodes as
   * informed by the parameters epsilon,delta and the robot's models.
   * @param S The current search tree.
   * @param R The robot to be planned for.
   * @param num The index of the robot being planned for.
   * @param y_T The predicted target trajectory.
   * @param eps The current value of epsilon.
   * @param del The current value of delta.
   * @param time_allocated_secs The total time allotted.
   * @param time_started The timer since the ARVI call started.
   * @param prune Flag indicating whether the search should prune or not. Set to false for Exhaustive Search.
   */
  void ImprovePath(std::vector<std::shared_ptr<nx::InfoStateSpace<state>>> &S,
                   const Robot<state> &R,
                   const std::vector<VectorXd> &y_T,
                   double eps,
                   double del,
                   double time_allocated_secs,
                   std::chrono::high_resolution_clock::time_point time_started,
                   bool prune = true) const;

  /**
   * Performs a quick (non-exhaustive) check if the matrix Sigma is algebraically redundant with respect to the matrices
   * in the vector Q.
   * @tparam state State-space of the Planner.
   * @param Sigma Matrix to check algebraic redundancy for.
   * @param Q The vector of matrices in the set.
   * @param epsilon Tolerance for checking algebraic redundancy.
   * @return True if Sigma is Alg-Red w.r.t. Q, false otherwise.
   */
  bool checkRedundancyQuick(const MatrixXd &Sigma,
                            const std::vector<MatrixXd> &Q,
                            double epsilon) const;

  /**
   * @brief Backout the optimal path from a completed search tree, writing the results into the passed in output object.
   * @param output The object to write the backed out path into.
   * @param search_tree The completed search tree from which the optimal path is extracted.
   */
  void backoutPath(PlannerOutput<state> &output,
                   const nx::Robot<state> &robot,
                   const std::shared_ptr<nx::InfoState<state>> &optimal_node) const;

  /**
   * @brief Writes the computed paths into the output log variables.
   * @param output The output to be logged into.
   * @param search_tree The completed search tree containing the optimal path.
   */
  void logPaths(PlannerOutput<state> &output,
                const nx::Robot<state> &robot,
                const InfoStateSpace<state> &search_tree,
                int t) const;



 public:

  /**
   * @brief Computes the value of the Heuristic function given the current information state, robot models,
   * remaining timesteps, and target state at the current time.
   * @param info_state Current Information State.
   * @param robot The robot planning for.
   * @param y The target state.
   * @param T The remaining timesteps in the planning process.
   * @return The value of the heuristic.
   */
  double computeHeuristic(const state & coord,
                          const MatrixXd &prior,
                          const Robot<state> &robot,
                          VectorXd y, int T) const;

}; // End Planner class.
} // End namespace


#endif //INFO_GATHERING_INFO_PLANNER_H
