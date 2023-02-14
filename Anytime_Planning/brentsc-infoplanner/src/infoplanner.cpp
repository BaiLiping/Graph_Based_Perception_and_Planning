
#include <igl/planning/infoplanner.h>
#include <igl/env/env_int.h>
#include <igl/robot.h>
#include <igl/estimation/multi_target_filter.h>
#include <igl/utils/utils_nx.h>
#include <igl/se3_pose.h>
#include <Eigen/StdVector>
/***********************************************************************************
 Information Acquisition Planner Implementations
************************************************************************************/
template<class state>
nx::PlannerOutput<state>
nx::InfoPlanner<state>::PlanARVI(const Robot<state> &robot,
                                 const int &T,
                                 const double &del,
                                 const double &epsilon,
                                 const double &allocated_time_secs,
                                 const int &debug,
                                 bool log) const {

  PlannerOutput<state> output; // This will be the Planner output
  state x0 = robot.getState();
  int start_idx = robot.env->state_to_idx(x0);

  // Begin Algorithm Clock
  auto time_started = std::chrono::high_resolution_clock::now();

  std::vector<VectorXd> y_T = robot.tmm->predictTargetState(T);    // Predict target trajectory for t = 1,...T-1
  MatrixXd Sigma = robot.tmm->getCovarianceMatrix();    // Get initial Sigma at t=0.

  // Initialize S0
  std::shared_ptr<InfoStateSpace<state>> S0(new InfoStateSpace<state>());
  // Initialize Start Node
  std::shared_ptr<InfoState<state>> currNode_pt(new InfoState<state>(start_idx, x0, Sigma, 0, true));

  currNode_pt->g = cost_function->computeCost(currNode_pt->Sigma);
  currNode_pt->is_open = true;
  currNode_pt->state_list->push(std::make_pair(currNode_pt->g, currNode_pt));

  // Insert type into the Map
  (currNode_pt->heapkey) = S0->pq.push(std::make_pair(currNode_pt->g, currNode_pt));
  S0->hm.insert(std::make_pair(start_idx, currNode_pt));

  // Initialize the list of state spaces
  std::vector<std::shared_ptr<nx::InfoStateSpace<state>>> S;
  S.push_back(S0);
  for (int t = 1; t <= T; t++) {
    std::shared_ptr<InfoStateSpace<state>> S_t(new InfoStateSpace<state>());
    S.push_back(S_t);
  }

  // Calculate initial greedy solution
  double delta = std::numeric_limits<double>::infinity();
  double eps = std::numeric_limits<double>::infinity();
  ImprovePath(S, robot, y_T, eps, delta, std::numeric_limits<double>::infinity(),
              time_started);
  //std::cout << "Greedy cost is: " << S[T]->pq.top().second->g << std::endl; // Prints cumulative cost
  delta = del;
  eps = epsilon;
  std::chrono::high_resolution_clock::time_point loop_time;

  // Loop as long as there is time remaining
  while (toc(time_started) < allocated_time_secs) {
    ImprovePath(S, robot, y_T, eps, delta, allocated_time_secs,
                time_started);
    loop_time = std::chrono::high_resolution_clock::now();
    delta = std::max(delta - .05, 0.0); // Ensure delta doesn't go below zero.
    eps = std::max(eps - .5, 0.0); // Ensure epsilon doesn't go below zero.
    if (eps == 0.0 && del == 0.0) // No further improvements will be made.
      break;
  }

  if (S[T]->pq.size() == 0) {
    std::cout << "No feasible path in desired Horizon.\n";
    S[T] = S[1];
  }

  //print the solution cost and eps bound
  if (debug) {    // Compute final size of tree...
    int num_expansions = 0;
    int opened_states = 0;
    for (int t = 0; t <= T; t++) {
      for (auto it = S[t]->pq.begin(); it != S[t]->pq.end(); ++it) {
        for (auto jt = it->second->state_list->begin(); jt != it->second->state_list->end(); ++jt) {
          std::shared_ptr<InfoState<state>> currNode_pt = (*jt).second;
//          if (currNode_pt->is_open)
          opened_states++;
          if (currNode_pt->is_closed)
            num_expansions++;

        }
      }
    }

    std::cout << "ARVI Final parameters: "
              << "T= " << T << " "
              << "num expansions" << num_expansions << " "
              << "open nodes " << opened_states << " "
              << "leaf nodes = " << S[T]->pq.size() << " "
              << "eps= " << eps << " "
              << "del = " << delta << " "
              << "optimal cost = " << S[T]->pq.top().second->g << " "
              << "time(total)= " << toc(time_started) << std::endl;
  }

  // Write the Output
  currNode_pt = S[T]->pq.top().second;   // Get Optimal Node
  backoutPath(output, robot, currNode_pt); // Backout Optimal Path and action indices.
  output.outputCovariance = currNode_pt->Sigma;
  output.target_path = y_T;
  if (log)
    logPaths(output, robot, *S[T], T); // Log all output paths only if requested.
  return output;
}

template<class state>
nx::PlannerOutput<state> nx::InfoPlanner<state>::PlanRVI(const nx::Robot<state> &robot,
                                                         const int &T,
                                                         const double &del,
                                                         const double &epsilon,
                                                         const int &debug,
                                                         bool log) const {

  PlannerOutput<state> output; // This will be the Planner output
  state x0 = robot.getState();
  int start_idx = robot.env->state_to_idx(x0);

  // Begin Algorithm Clock
  auto time_started = std::chrono::high_resolution_clock::now();

  std::vector<VectorXd> y_T = robot.tmm->predictTargetState(T);    // Predict target trajectory for t = 1,...T-1
  MatrixXd Sigma = robot.tmm->getCovarianceMatrix();    // Get initial Sigma at t=0.

  // Initialize S0
  std::shared_ptr<InfoStateSpace<state>> S0(new InfoStateSpace<state>());
  // Initialize Start Node
  std::shared_ptr<InfoState<state>> currNode_pt(new InfoState<state>(start_idx, x0, Sigma, 0, true));

  currNode_pt->g = cost_function->computeCost(currNode_pt->Sigma);
  currNode_pt->is_open = true;
  currNode_pt->state_list->push(std::make_pair(currNode_pt->g, currNode_pt));

  // Insert type into the Map
  (currNode_pt->heapkey) = S0->pq.push(std::make_pair(currNode_pt->g, currNode_pt));
  S0->hm.insert(std::make_pair(start_idx, currNode_pt));

  // Initialize the list of state spaces
  std::vector<std::shared_ptr<nx::InfoStateSpace<state>>> S;
  S.push_back(S0);
  for (int t = 1; t <= T; t++) {
    std::shared_ptr<InfoStateSpace<state>> S_t(new InfoStateSpace<state>());
    S.push_back(S_t);
  }

  // Main Planning Call.
  ImprovePath(S, robot, y_T, epsilon, del, std::numeric_limits<double>::infinity(), nx::tic());

  if (S[T]->pq.size() == 0) {
    std::cout << "No feasible path in desired Horizon.\n";
    S[T] = S[1];
  }

  // Compute final size of tree...
  if (debug) {
    int count = 0;
    int true_count = 0;
    for (auto it = S[T]->pq.begin(); it != S[T]->pq.end(); ++it) {
      count++;
      true_count += (*it).second->state_list->size();
    }
    if (debug)
      std::cout << "RVI Final parameters: "
                << "T= " << T << " "
                << "tree size = " << S[T]->pq.size() << " "
                << "eps= " << epsilon << " "
                << "del = " << del << " "
                << "optimal cost = " << S[T]->pq.top().second->g << " "
                << "time(total)= " << toc(time_started) << std::endl;
  }
  // Write the Output
  currNode_pt = S[T]->pq.top().second;   // Get Optimal Node
  backoutPath(output, robot, currNode_pt); // Backout Optimal Path and action indices.
  output.outputCovariance = currNode_pt->Sigma;
  output.target_path = y_T;
  if (log)
    logPaths(output, robot, *S[T], T); // Log all output paths only if requested.
  return output;
}

template<class state>
nx::PlannerOutput<state> nx::InfoPlanner<state>::PlanFVI(const nx::Robot<state> &robot,
                                                         const int &T,
                                                         const int &debug,
                                                         bool log) const {

  PlannerOutput<state> output; // This will be the Planner output
  state x0 = robot.getState();
  int start_idx = robot.env->state_to_idx(x0);

  // Begin Algorithm Clock
  auto time_started = std::chrono::high_resolution_clock::now();

  std::vector<VectorXd> y_T = robot.tmm->predictTargetState(T);    // Predict target trajectory for t = 1,...T-1
  MatrixXd Sigma = robot.tmm->getCovarianceMatrix();    // Get initial Sigma at t=0.

  // Initialize S0
  std::shared_ptr<InfoStateSpace<state>> S0(new InfoStateSpace<state>());
  // Initialize Start Node
  std::shared_ptr<InfoState<state>> currNode_pt(new InfoState<state>(start_idx, x0, Sigma, 0, true));

  currNode_pt->g = cost_function->computeCost(currNode_pt->Sigma);
  currNode_pt->is_open = true;
  currNode_pt->state_list->push(std::make_pair(currNode_pt->g, currNode_pt));

  // Insert type into the Map
  (currNode_pt->heapkey) = S0->pq.push(std::make_pair(currNode_pt->g, currNode_pt));
  S0->hm.insert(std::make_pair(start_idx, currNode_pt));

  // Initialize the list of state spaces
  std::vector<std::shared_ptr<nx::InfoStateSpace<state>>> S;
  S.push_back(S0);
  for (int t = 1; t <= T; t++) {
    std::shared_ptr<InfoStateSpace<state>> S_t(new InfoStateSpace<state>());
    S.push_back(S_t);
  }

  // Main Planning Call.
  ImprovePath(S, robot, y_T, 0, 0, std::numeric_limits<double>::infinity(), nx::tic(), false);

  if (S[T]->pq.size() == 0) {
    std::cout << "No feasible path in desired Horizon.\n";
    S[T] = S[1];
  }

  // Compute final size of tree...
  if (debug) {
    int count = 0;
    int true_count = 0;
    for (auto it = S[T]->pq.begin(); it != S[T]->pq.end(); ++it) {
      count++;
      true_count += (*it).second->state_list->size();
    }
    if (debug)
      std::cout << "FVI Final parameters: "
                << "T= " << T << " "
                << "tree size = " << S[T]->pq.size() << " "
                << "optimal cost = " << S[T]->pq.top().second->g << " "
                << "time(total)= " << toc(time_started) << std::endl;
  }

  // Write the Output
  currNode_pt = S[T]->pq.top().second;   // Get Optimal Node
  backoutPath(output, robot, currNode_pt); // Backout Optimal Path and action indices.
  output.outputCovariance = currNode_pt->Sigma;
  output.target_path = y_T;
  if (log)
      logPaths(output, robot, *S[T], T); // Log all output paths only if requested.
  return output;
}

template<class state>
void
nx::InfoPlanner<state>::ImprovePath(std::vector<std::shared_ptr<nx::InfoStateSpace<state>>> &S,
                                    const Robot<state> &R,
                                    const std::vector<VectorXd> &y_T,
                                    double eps,
                                    double del,
                                    double time_allocated_secs,
                                    std::chrono::high_resolution_clock::time_point time_started,
                                    bool prune) const {

  std::shared_ptr<InfoState<state>> currNode_pt;
  int T = (int) S.size() - 1;
  int rvi_nodes_pruned = 0;

  for (int t = 1; t <= T; t++) {
    //std::cout <<"Search tree at time : " << t << " eps="<< eps <<" del=" << del << " size="<< S[t-1]->hm.size()<< std::endl;
    if (toc(time_started) >= time_allocated_secs)     // Check during the loop if the function should exit.
      return;
    // For all (x,Sigma) in O_{t-1}
//    for (auto it = S[t - 1]->hm.begin(); it != S[t - 1]->hm.end(); ++it) {
//      for (auto jt = it->second->state_list->begin(); jt != it->second->state_list->end(); ++jt) {
    for (auto it = S[t - 1]->pq.begin(); it != S[t - 1]->pq.end(); ++it) {
      for (auto jt = it->second->state_list->begin(); jt != it->second->state_list->end(); ++jt) {

        currNode_pt = jt->second;
        // Search through O[t-1]
        if (currNode_pt->is_open && !currNode_pt->is_closed) {
          // std::cout << "Getting successors of open node: " << env->state_to_vector(currNode_pt->coord) << std::endl;
          std::vector<state> succ_coord;
          std::vector<double> succ_cost;
          std::vector<int> succ_act_idx;
          R.env->get_succ(currNode_pt->coord, succ_coord, succ_cost, succ_act_idx);
          // For all u in U... Process successors
          for (unsigned s = 0; s < succ_coord.size(); ++s) {
            std::shared_ptr<InfoState<state>> succNode_pt;
            state x_t = succ_coord[s];
            std::shared_ptr<InfoState<state>> best_pt;
            double min_cov = std::numeric_limits<double>::infinity();
            try { // TODO Replace this try - catch.
              // This means the state exists already, but we can reach it in a new way.
              succNode_pt = S[t]->hm.at(R.env->state_to_idx(succ_coord[s]));
              MatrixXd Sigma_t = MultiTargetFilter<state>::MultiTargetKFCovariance(R, x_t, y_T[t], currNode_pt->Sigma);

              const std::shared_ptr<InfoState<state>> temp_Node( // Construct a new Information State
                  new InfoState<state>(R.env->state_to_idx(succ_coord[s]),
                                       succ_coord[s], Sigma_t, t, false));
              temp_Node->g = currNode_pt->g + cost_function->computeCost(Sigma_t);
              temp_Node->parent = (*jt).second;
              temp_Node->parent_action_id = succ_act_idx[s];
              temp_Node->state_list = succNode_pt->state_list; // Ensures state_list is correct.
              // Now add temp_node to the list
              succNode_pt->state_list->push(
                  std::make_pair(temp_Node->g, temp_Node));
              if (temp_Node->g < min_cov) { // Here we also ensure that the state list order is maintained.
                best_pt = temp_Node;
                min_cov = temp_Node->g;
                succNode_pt.swap(best_pt);
                S[t]->pq.decrease(S[t]->hm[R.env->state_to_idx((succ_coord[s]))]->heapkey);
              }
            } // End Existing Spatial State.
            catch (std::out_of_range e) {
              // This means the state is completely new.
              MatrixXd Sigma_t = MultiTargetFilter<state>::MultiTargetKFCovariance(R, x_t, y_T[t], currNode_pt->Sigma);

              std::shared_ptr<InfoState<state>> temp_pt( // Construct a new Information State
                  new InfoState<state>(R.env->state_to_idx(succ_coord[s]),
                                       succ_coord[s], Sigma_t, t, true), D<state>());
              succNode_pt = temp_pt;
              succNode_pt->g = currNode_pt->g + cost_function->computeCost(Sigma_t);
              succNode_pt->Sigma = Sigma_t;

              succNode_pt->parent = currNode_pt;
              succNode_pt->parent_action_id = succ_act_idx[s];
              succNode_pt->state_list->push(std::make_pair(succNode_pt->g, succNode_pt));
              S[t]->hm.insert(std::make_pair(R.env->state_to_idx(succ_coord[s]), succNode_pt));
              succNode_pt->heapkey = S[t]->pq.push(
                  std::make_pair(succNode_pt->g, succNode_pt));
            } // End New Spatial State
          } // End get_Successors
          currNode_pt->is_closed = true; // Close the expanded node.
        } // End Open / Closed check.
      }
    } // End forward simulation.

    if (!prune) { // For forward Value Iteration we must mark all Nodes as Open.
      for (auto it = S[t]->hm.begin(); it != S[t]->hm.end(); ++it)   // Loop over all States in S_t, but not O_t.
        for (auto qt = (*it).second->state_list->begin(); qt != (*it).second->state_list->end(); ++qt)
          (*qt).second->is_open = true;
    }

    if (!S[t]->pq.empty() && t < T && prune) { // Begin prune tree operations if possible.
      const std::shared_ptr<InfoState<state>> S_1 = S[t]->pq.top().second;
      S[t]->pq.top().second->is_open = true;        // Top element is always Open
      for (auto it = S[t]->hm.begin(); it != S[t]->hm.end(); ++it) {  // Loop over all States in S_t, but not O_t.
        for (auto qt = (*it).second->state_list->begin(); qt != (*it).second->state_list->end(); ++qt) {
          if (toc(time_started) >= time_allocated_secs)     // Check during the loop if the function should exit.
            return;
          currNode_pt = (*qt).second;
          // If the node is in S_t, but not O_t, check if currNode_pt should be added to O_t by checking Redundancy.
          if (!currNode_pt->is_open) {
            state x_t = currNode_pt->coord;
            MatrixXd cov = currNode_pt->Sigma;
            std::vector<MatrixXd> Q;
            // Find all Nodes in O_t which d-cross x_t:
            for (auto jt = S[t]->hm.begin(); jt != S[t]->hm.end(); ++jt) {
              for (auto mt = (*jt).second->state_list->begin(); mt != (*jt).second->state_list->end(); ++mt) {
                const std::shared_ptr<InfoState<state>> searchNode_pt = mt->second;
                // States are in O_t
                if (searchNode_pt->is_open) {
                  state x_t_prime = searchNode_pt->coord;
                  MatrixXd Sigma_t_prime = searchNode_pt->Sigma;
                  double metric = R.env->ComputeStateMetric(x_t, x_t_prime);
                  // If del-cross, we need to check the covariance for epsilon redundancy
                  if (metric <= del) {
                    Q.push_back(Sigma_t_prime);
                  }
                }
              }
            }
            bool is_redundant =
                (eps == std::numeric_limits<double>::infinity()) ?
                true : checkRedundancyQuick(cov, Q, eps);
            // This means we need to keep the state open.
            if (Q.empty() || !is_redundant)
              currNode_pt->is_open = true;
          } // End Redundancy Check
        }
      } // End Loop Over Nodes
    } // End Tree Pruning
  } // End Iteration t
} // End Function

template<class state>
bool nx::InfoPlanner<state>::checkRedundancyQuick(const MatrixXd &Sigma,
                                                  const std::vector<MatrixXd> &Q,
                                                  double epsilon) const {
  // Quick check verifies only K PSD inequalities instead of the full LMI
  bool redundant = false;
  for (int i = 0; i < Q.size() && !redundant; i++) {
    MatrixXd Sigma_i = Q[i];
    MatrixXd check_posdef = Sigma - Sigma_i + epsilon * MatrixXd::Identity(Sigma.rows(), Sigma.cols());
    redundant = (check_posdef.diagonal().array() >= 0.0).all();
    //redundant = (check_posdef.determinant() > 0.0);
  }
  return redundant;
}

template<class state>
void nx::InfoPlanner<state>::backoutPath(nx::PlannerOutput<state> &output,
                                         const nx::Robot<state> &robot,
                                         const std::shared_ptr<nx::InfoState<state>> &optimal_node) const {
  auto currNode_pt = optimal_node;

  while (currNode_pt->parent) {
    output.action_idx.push_back(currNode_pt->parent_action_id); // Write current Action
    output.observation_points.push_back(currNode_pt->coord); // Write Current Observation point.
    currNode_pt = currNode_pt->parent;
    std::vector<state> next_micro; // Write Micro-states for dense path.
    robot.env->forward_action(currNode_pt->coord, output.action_idx.back(), next_micro);
    for (typename std::vector<state>::reverse_iterator it = next_micro.rbegin();
         it != next_micro.rend(); ++it)
      output.path.push_front(*it);
  }
  //path.push_front(currNode_pt->coord);
}

template<class state>
void nx::InfoPlanner<state>::logPaths(nx::PlannerOutput<state> &output,
                                      const nx::Robot<state> &robot,
                                      const nx::InfoStateSpace<state> &search_tree, int t) const {
  // Iterate over all terminal Nodes and log their corresponding paths.
  double cost = 0;
  unsigned nodes = 0; // Total number of nodes
  for (const auto &s : search_tree.hm) {
    for (const auto &s2 : *s.second->state_list) {
//      if (s2.second->t > 0) {
//      }
      // Add states to opened node list.
      output.opened_list[s2.second->iteration_opened].push_back(s2.second->coord);

      // Add states to closed node list
      if (s2.second->iteration_closed)
        output.closed_list[s2.second->iteration_closed] = s2.second->coord;

      // Add path if it is a Leaf Node
      if (s2.second->t == t) {
        PlannerOutput<state> path_result;
        backoutPath(path_result, robot, s2.second);
        output.all_paths.push_back(path_result.path);
        cost += s2.second->g / s2.second->t;
        nodes++;
      }
    }
  }
  // Compute the average cost per node in the search tree.
  std::cout <<"Total cost contained in tree: " << cost << std::endl;
  std::cout <<"Total nodes contained in tree: " << nodes << std::endl;
  output.cost_per_node = cost / nodes;
}
template<class state>
nx::PlannerOutput<state> nx::InfoPlanner<state>::PlanAstar(const nx::Robot<state> &robot,
                                                           const int &T,
                                                           const double &epsilon,
                                                           const int &debug,
                                                           bool log) const {

  PlannerOutput<state> output; // This will be the Planner output
  state x0 = robot.getState();
  int start_idx = robot.env->state_to_idx(x0);

  // Begin Algorithm Clock
  auto time_started = std::chrono::high_resolution_clock::now();

  std::vector<VectorXd> y_T = robot.tmm->predictTargetState(T);    // Predict target trajectory for t = 1,...T-1
  MatrixXd Sigma = robot.tmm->getCovarianceMatrix();    // Get initial Sigma at t=0.

  // Initialize S0
  std::shared_ptr<InfoStateSpace<state>> S0(new InfoStateSpace<state>());
  // Initialize Start Node
  std::shared_ptr<InfoState<state>> currNode_pt(new InfoState<state>(start_idx, x0, Sigma, 0, true));

  currNode_pt->g = cost_function->computeCost(currNode_pt->Sigma);
  currNode_pt->is_open = true;
  currNode_pt->state_list->push(std::make_pair(currNode_pt->g, currNode_pt));

  // Insert type into the Map
  (currNode_pt->heapkey) = S0->pq.push(std::make_pair(currNode_pt->g, currNode_pt));
  S0->hm.insert(std::make_pair(start_idx, currNode_pt));


  // Main Planning.
  bool done = false;
  int num_expands = 0;
  while (currNode_pt->t < T) // While not at goal
  {
    currNode_pt->iteration_closed = num_expands++;
    //    std::cout <<"Num expansions" << ++num_expands << std::endl;
    // Loop over successors of currNode_pt
    std::vector<state> succ_coord;
    std::vector<double> succ_cost;
    std::vector<int> succ_act_idx;
    robot.env->get_succ(currNode_pt->coord, succ_coord, succ_cost, succ_act_idx);
    // For all u in U... Process successors
    for (unsigned s = 0; s < succ_coord.size(); ++s) {
      int t = currNode_pt->t;
      state x_t = succ_coord[s];
      int lindix = robot.env->state_to_idx(succ_coord[s]);

      MatrixXd Sigma_t = MultiTargetFilter<state>::MultiTargetKFCovariance(robot, x_t, y_T[t], currNode_pt->Sigma);

      // TODO Check on prediction timestep correct
      const std::shared_ptr<InfoState<state>> succNode_pt( // Construct a new Information State
          new InfoState<state>(robot.env->state_to_idx(succ_coord[s]),
                               succ_coord[s], Sigma_t, t + 1, true));
      succNode_pt->g = currNode_pt->g + cost_function->computeCost(Sigma_t);
      succNode_pt->parent = currNode_pt;
      succNode_pt->parent_action_id = succ_act_idx[s];
      succNode_pt->state_list->push(std::make_pair(succNode_pt->g, succNode_pt));
      succNode_pt->iteration_opened = num_expands;
      succNode_pt->heapkey =
          S0->pq.push(std::make_pair(
              succNode_pt->g + epsilon * computeHeuristic(succ_coord[s], Sigma_t, robot, y_T[t], T - t),
              succNode_pt));
      if (S0->hm.count(lindix)) // Node exists in space.
      {
        S0->hm.at(lindix)->state_list->push(std::make_pair(succNode_pt->g, succNode_pt));
      } else {
        S0->hm.insert(std::make_pair(robot.env->state_to_idx(succ_coord[s]), succNode_pt));
      }
    }

    currNode_pt = S0->pq.top().second;
    S0->pq.pop();
  }

  // Compute final size of tree...
  if (debug) {
    int leaf_nodes = 0;
    int all_nodes = 0;
    for (auto it = S0->hm.begin(); it != S0->hm.end(); ++it) {
      for (auto jt = it->second->state_list->begin(); jt != it->second->state_list->end(); jt++) {
        if (jt->second->t == T)
          leaf_nodes++; // This is the count of leaf nodes.
        all_nodes++; // This is the count of opened states.
      }
    }

    std::cout << "Astar Final parameters: "
              << "T= " << T << " "
              << "num expansions = " << num_expands << " "
              << "open nodes = " << all_nodes << " "
              << "leaf nodes = " << leaf_nodes << " "
              << "optimal cost = " << currNode_pt->g << " "
              << "time(total)= " << toc(time_started) << std::endl;
  }

  // Write the Output
//  currNode_pt = S0->pq.top().second;   // Get Optimal Node
  backoutPath(output, robot, currNode_pt); // Backout Optimal Path and action indices.
  output.outputCovariance = currNode_pt->Sigma;
  output.target_path = y_T;
  if (log)
    logPaths(output, robot, *S0, T); // Log all output paths only if requested.
  return output;
}

template<class state>
double nx::InfoPlanner<state>::computeHeuristic(const state &coord,
                                                const MatrixXd &prior,
                                                const nx::Robot<state> &robot,
                                                VectorXd y,
                                                int T) const {
  double result = 0;
  int y_dim = prior.rows();
  for (int t = 1; t <= T; t++) {
    // Get Jacobians
    MatrixXd A(y_dim, y_dim);
    MatrixXd W(y_dim, y_dim);
    MatrixXd M = robot.sensor->maxSensorMatrix(coord, y, robot.tmm, t);
    robot.tmm->getJacobian(A, W);
    MatrixXd Sigma_pred = A * prior * A.transpose() + W;
    MatrixXd Sigma_update = (Sigma_pred.inverse() + M).inverse();
    y = A * y;
    result += cost_function->computeCost(Sigma_update);
  }
  return result;
}

template
class nx::InfoPlanner<nx::SE3Pose>;

template
class nx::PlannerOutput<nx::SE3Pose>;
