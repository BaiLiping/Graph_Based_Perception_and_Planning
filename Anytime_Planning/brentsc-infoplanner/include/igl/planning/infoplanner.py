import igl
import numpy as np
from queue import PriorityQueue
from igl.env.env_int import EnvInt
from igl.robot import Robot
from igl.estimation.multi_target_filter import MultiTargetFilter
from igl.utils.utils_nx import tic, toc
from igl.se3_pose import SE3Pose
import heapq
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict, Any
import time


import numpy as np
from heapq import heappush, heappop
from collections import defaultdict

class InfoStateSpace:
    def __init__(self):
        self.pq = []
        self.hm = {}

class InfoState:
    def __init__(self, idx, coord, sigma, t, is_open):
        self.idx = idx
        self.coord = coord
        self.sigma = sigma
        self.t = t
        self.is_open = is_open
        self.is_closed = False
        self.parent = None
        self.parent_action_id = None
        self.state_list = []

class InfoPlanner:
    def __init__(self):
        pass
    
    def plan_rvi(self, robot, T, epsilon, del_, debug=False, log=False):
        # Initialize output, input state, and S0
        output = PlannerOutput()
        x0 = robot.get_state()
        start_idx = robot.env.state_to_idx(x0)
        sigma = robot.tmm.get_covariance_matrix()
        S0 = InfoStateSpace()
        start_node = InfoState(start_idx, x0, sigma, 0, True)
        start_node.g = cost_function.compute_cost(sigma)
        start_node.state_list.append((start_node.g, start_node))
        S0.pq.append((start_node.g, start_node))
        S0.hm[start_idx] = start_node
        
        # Initialize the list of state spaces
        S = [S0]
        for t in range(1, T + 1):
            S_t = InfoStateSpace()
            S.append(S_t)
        
        # Main Planning Call
        y_T = robot.tmm.predict_target_state(T)
        self.improve_path(S, robot, y_T, epsilon, del_, np.inf)
        
        if not S[T].pq:
            print("No feasible path in desired Horizon.")
            S[T] = S[1]
        
        # Compute final size of tree
        if debug:
            count = 0
            true_count = 0
            for _, state in S[T].hm.items():
                count += 1
                true_count += len(state.state_list)
            print(f"RVI Final parameters: T = {T}, tree size = {count}, eps = {epsilon}, del = {del_}, optimal cost = {S[T].pq[0][1].g}")
        
        # Write the Output
        curr_node = S[T].pq[0][1]  # Get Optimal Node
        self.backout_path(output, robot, curr_node)  # Backout Optimal Path and action indices
        output.output_covariance = curr_node.sigma
        output.target_path = y_T
        if log:
            self.log_paths(output, robot, S[T], T)
        
        return output
    
    def plan_fvi(self, robot, T, epsilon, del_, debug=False, log=False):
        # Initialize output, input state, and S0
        output = PlannerOutput()
        x0 = robot.get_state()
        start_idx = robot.env.state_to_idx(x0)
        sigma = robot.tmm.get_covariance_matrix()
        S0 = InfoStateSpace()
        start_node = InfoState(start_idx, x0, sigma, 0, True)
        start_node.g = cost_function.compute_cost(sigma)
        start_node.state_list.append((start_node.g, start_node))
        S0.pq.append((start_node.g, start_node))
        S0.hm[start_idx] = start_node
    
        # Initialize the list of state spaces
        S = [S0]
        for t in range(1, T + 1):
            S_t = InfoStateSpace()
            S.append(S_t)
    
        # Main Planning Call
        y_T = robot.tmm.predict_target_state(T)
        t = 1
        while t <= T:
            self.improve_path(S, robot, y_T, epsilon, del_, t)
            if not S[t].pq:
                t -= 1
                if t == 0:
                    print("No feasible path in desired Horizon.")
                    S[T] = S[1]
                    break
            else:
                t += 1
    
        # Compute final size of tree
        if debug:
            count = 0
            true_count = 0
            for _, state in S[T].hm.items():
                count += 1
                true_count += len(state.state_list)
            print(f"FVI Final parameters: T = {T}, tree size = {count}, eps = {epsilon}, del = {del_}, optimal cost = {S[T].pq[0][1].g}")
    
        # Write the Output
        curr_node = S[T].pq[0][1]  # Get Optimal Node
        self.backout_path(output, robot, curr_node)  # Backout Optimal Path and action indices
        output.output_covariance = curr_node.sigma
        output.target_path = y_T
        if log:
            self.log_paths(output, robot, S[T], T)
    
        return output


    def improve_path(self, S, robot, y_T, epsilon, del_, t):
        S_t = S[t]
        S_t_minus_1 = S[t - 1]
    
        while S_t_minus_1.pq:
            # Retrieve and remove the best node from S_t_minus_1
            g_t_minus_1, node_t_minus_1 = heapq.heappop(S_t_minus_1.pq)
            idx_t_minus_1 = node_t_minus_1.idx
    
            for a in range(robot.env.num_actions):
                # Calculate next state, state index, and transition probability
                x_t, p_t = robot.tmm.transition_model(idx_t_minus_1, a)
                idx_t = robot.env.state_to_idx(x_t)
    
                # Calculate the cost of the current state and action
                cost_t = self.cost_function.compute_cost(node_t_minus_1.sigma, a, p_t)
    
                # Check whether the next state is visited before
                if idx_t in S_t.hm:
                    node_t = S_t.hm[idx_t]
                else:
                    node_t = InfoState(idx_t, x_t, None, 0, False)
                    S_t.hm[idx_t] = node_t
    
                # Calculate new sigma and g
                new_sigma = robot.tmm.update_covariance(node_t_minus_1.sigma, a, p_t)
                new_g = g_t_minus_1 + cost_t + self.cost_function.compute_cost(new_sigma)
    
                if not node_t.visited or new_g < node_t.g - epsilon:
                    # Update node_t's information
                    node_t.visited = True
                    node_t.sigma = new_sigma
                    node_t.g = new_g
                    node_t.state_list.append((new_g, node_t_minus_1))
    
                    # Update the priority queue
                    heapq.heappush(S_t.pq, (new_g + self.discount_factor * self.heuristic_function(y_T, x_t, del_, t), node_t))


    def compute_heuristic(coord: np.ndarray, prior: np.ndarray, robot: Any, y: np.ndarray, T: int) -> float:
        result = 0
        y_dim = prior.shape[0]
        
        for t in range(1, T + 1):
            # Get Jacobians
            A, W = robot.tmm.get_jacobian()
            M = robot.sensor.max_sensor_matrix(coord, y, robot.tmm, t)
            Sigma_pred = A @ prior @ A.T + W
            Sigma_update = np.linalg.inv(Sigma_pred) + M
            Sigma_update = np.linalg.inv(Sigma_update)
            
            y = A @ y
            result += cost_function.compute_cost(Sigma_update)
            
        return result
    
    def plan_astar(robot: Any, T: int, epsilon: float, debug: bool = False, log: bool = False) -> Dict[str, Any]:
        output = defaultdict(list)  # This will be the Planner output
        x0 = robot.get_state()
        start_idx = robot.env.state_to_idx(x0)
    
        # Begin Algorithm Clock
        time_started = time.perf_counter()
    
        y_T = robot.tmm.predict_target_state(T)  # Predict target trajectory for t = 1,...T-1
        Sigma = robot.tmm.get_covariance_matrix()  # Get initial Sigma at t=0.
    
        # Initialize S0
        S0 = {
            "pq": [],
            "hm": {}
        }
    
        # Initialize Start Node
        curr_node = {
            "idx": start_idx,
            "coord": x0,
            "Sigma": Sigma,
            "t": 0,
            "g": cost_function.compute_cost(Sigma)
        }
        heapq.heappush(S0["pq"], (curr_node["g"], curr_node))
        S0["hm"][start_idx] = curr_node
    
        num_expands = 0
        while curr_node["t"] < T:  # While not at goal
            curr_node["iteration_closed"] = num_expands
            num_expands += 1
    
            # Loop over successors of curr_node
            succ_coord, succ_cost, succ_act_idx = robot.env.get_succ(curr_node["coord"])
    
            for s in range(len(succ_coord)):
                t = curr_node["t"]
                x_t = succ_coord[s]
                lindix = robot.env.state_to_idx(succ_coord[s])
    
                Sigma_t = MultiTargetFilter.multi_target_kf_covariance(robot, x_t, y_T[t], curr_node["Sigma"])
    
                # Construct a new Information State
                succ_node = {
                    "idx": lindix,
                    "coord": x_t,
                    "Sigma": Sigma_t,
                    "t": t + 1,
                    "g": curr_node["g"] + cost_function.compute_cost(Sigma_t),
                    "parent": curr_node,
                    "parent_action_id": succ_act_idx[s]
                }
    
                # Compute heuristic
                h = compute_heuristic(x_t, Sigma_t, robot, y_T[t], T - t)
    
                heapq.heappush(S0["pq"], (succ_node["g"] + epsilon * h, succ_node))
    
                if lindix in S0["hm"]:
                    S0["hm"][lindix].append(succ_node)
                else:
                    S0["hm"][lindix] = [succ_node]
    
            curr_node = heapq.heappop(S0["pq"])[1]
    
        # Write the Output
        backout_path(output, robot, curr_node)  # Backout Optimal Path and action indices.
        output["outputCovariance"] = curr_node["Sigma"]
        output["target_path"] = y_T
    
        if log:
            log_paths(output, robot, S0["hm"], T)  # Log all output paths only if requested.
    
        return output
    
    def log_paths(output, robot, search_tree, t):
        total_cost = 0
        total_nodes = 0
    
        for s in search_tree.hm.values():
            for s2 in s.state_list.values():
                # Add states to opened node list.
                output['opened_list'][s2.iteration_opened].append(s2.coord)
    
                # Add states to closed node list
                if s2.iteration_closed:
                    output['closed_list'][s2.iteration_closed] = s2.coord
    
                # Add path if it is a Leaf Node
                if s2.t == t:
                    path_result = {
                        'path': [],
                    }
                    backout_path_result = backout_path(robot, s2)
                    path_result['path'] = backout_path_result['path']
                    output['all_paths'].append(path_result['path'])
                    total_cost += s2.g / s2.t
                    total_nodes += 1
    
        print("Total cost contained in tree:", total_cost)
        print("Total nodes contained in tree:", total_nodes)
        output['cost_per_node'] = total_cost / total_nodes


    def backout_path(robot, optimal_node):
        output = {
            'action_idx': [],
            'observation_points': [],
            'path': []
        }
    
        curr_node = optimal_node
        while curr_node.parent is not None:
            output['action_idx'].append(curr_node.parent_action_id)
            output['observation_points'].append(curr_node.coord)
            curr_node = curr_node.parent
            next_micro = []
            robot.env.forward_action(curr_node.coord, output['action_idx'][-1], next_micro)
            output['path'][0:0] = next_micro[::-1]
    
        return output

    def check_redundancy_quick(Sigma, Q, epsilon):
        redundant = False
        for i in range(len(Q)):
            Sigma_i = Q[i]
            check_posdef = Sigma - Sigma_i + epsilon * np.identity(Sigma.shape[0])
            redundant = np.all(check_posdef.diagonal() >= 0.0)
            if redundant:
                break
        return redundant