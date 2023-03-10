import igl
import numpy as np
from queue import PriorityQueue
from igl.env.env_int import EnvInt
from igl.robot import Robot
from igl.estimation.multi_target_filter import MultiTargetFilter
from igl.utils.utils_nx import tic, toc
from igl.se3_pose import SE3Pose

class InfoStateSpace:
    def __init__(self):
        self.pq = PriorityQueue()
        self.hm = {}

class InfoState:
    def __init__(self, idx, x, Sigma, g, is_open):
        self.idx = idx
        self.x = x
        self.Sigma = Sigma
        self.g = g
        self.is_open = is_open
        self.is_closed = False
        self.heapkey = None
        self.state_list = PriorityQueue()

class InfoPlanner:
    def __init__(self, cost_function):
        self.cost_function = cost_function

    def PlanARVI(self, robot, T, del_, epsilon, allocated_time_secs, debug, log):
        output = igl.PlannerOutput() # This will be the Planner output
        x0 = robot.getState()
        start_idx = robot.env.state_to_idx(x0)

        # Begin Algorithm Clock
        time_started = tic()

        y_T = robot.tmm.predictTargetState(T) # Predict target trajectory for t = 1,...T-1
        Sigma = robot.tmm.getCovarianceMatrix() # Get initial Sigma at t=0.

        # Initialize S0
        S0 = InfoStateSpace()
        # Initialize Start Node
        currNode_pt = InfoState(start_idx, x0, Sigma, 0, True)

        currNode_pt.g = self.cost_function.computeCost(currNode_pt.Sigma)
        currNode_pt.is_open = True
        currNode_pt.state_list.put((currNode_pt.g, currNode_pt))

        # Insert type into the Map
        currNode_pt.heapkey = S0.pq.put((currNode_pt.g, currNode_pt))
        S0.hm[start_idx] = currNode_pt

        # Initialize the list of state spaces
        S = [S0]
        for t in range(1, T+1):
            S_t = InfoStateSpace()
            S.append(S_t)

        # Calculate initial greedy solution
        delta = np.inf
        eps = np.inf
        ImprovePath(S, robot, y_T, eps, delta, np.inf, time_started)
        #print("Greedy cost is: ", S[T].pq.top().second.g) # Prints cumulative cost
        delta = del_
        eps = epsilon
        loop_time = tic()

        # Loop as long as there is time remaining
        while toc(time_started) < allocated_time_secs:
            ImprovePath(S, robot, y_T, eps, delta, allocated_time_secs, time_started)
            loop_time = tic()
            delta = max(delta - .05, 0.0) # Ensure delta doesn't go below zero.
            eps = max(eps - .5, 0.0) # Ensure epsilon doesn't go below zero.
            if eps == 0.0 and del_ == 0.0: # No further improvements will be made.
                break

        if S[T].pq.qsize() == 0:
            print("No feasible path in desired Horizon.\n")
            S[T] = S[1]

        #print the solution cost and eps bound
        if debug:
            num_expansions = 0
            opened_states = 0
            for t in range(T+1):
                for _,currNode_pt in S[t].pq.queue:
                    for _, child in currNode_pt.state_list.queue:
                        if child.is_open:
                            opened_states += 1
                        if child.is_closed:
                            num_expansions += 1
    
            print("ARVI Final parameters: T={}, num expansions={}, open nodes={}, leaf nodes={}, eps={}, del={}, optimal cost={}, time(total)={}".format(
                T, num_expansions, opened_states, S[T].pq.qsize(), eps, delta, S[T].pq.queue[0][0], toc(time_started)
            ))
    
        # Write the Output
        currNode_pt = S[T].pq.queue[0][1]   # Get Optimal Node
        backoutPath(output, robot, currNode_pt) # Backout Optimal Path and action indices.
        output.outputCovariance = currNode_pt.Sigma
        output.target_path = y_T
        if log:
            logPaths(output, robot, S[T], T) # Log all output paths only if requested.
        return output

    def PlanRVI(self, robot, T, del_, epsilon, debug, log):
        output = igl.PlannerOutput() # This will be the Planner output
        x0 = robot.getState()
        start_idx = robot.env.state_to_idx(x0)
    
        # Begin Algorithm Clock
        time_started = tic()
    
        y_T = robot.tmm.predictTargetState(T) # Predict target trajectory for t = 1,...T-1
        Sigma = robot.tmm.getCovarianceMatrix() # Get initial Sigma at t=0.
    
        # Initialize S0
        S0 = InfoStateSpace()
        # Initialize Start Node
        currNode_pt = InfoState(start_idx, x0, Sigma, 0, True)
    
        currNode_pt.g = self.cost_function.computeCost(currNode_pt.Sigma)
        currNode_pt.is_open = True
        currNode_pt.state_list.put((currNode_pt.g, currNode_pt))
    
        # Insert type into the Map
        currNode_pt.heapkey = S0.pq.put((currNode_pt.g, currNode_pt))
        S0.hm[start_idx] = currNode_pt
    
        # Initialize the list of state spaces
        S = [S0]
        for t in range(1, T+1):
            S_t = InfoStateSpace()
            S.append(S_t)
    
        # Main Planning Call.
        ImprovePath(S, robot, y_T, epsilon, del_, np.inf, tic())
    
        if S[T].pq.qsize() == 0:
            print("No feasible path in desired Horizon.\n")
            S[T] = S[1]
    
        # Compute final size of tree...
        if debug:
            count = 0
            true_count = 0
            for _, currNode_pt in S[T].pq.queue:
                count += 1
                true_count += currNode_pt.state_list.qsize()
    
            print("RVI Final parameters: T={}, tree size={}, eps={}, del={}, optimal cost={}, time(total)={}".format(
                T, S[T].pq.qsize(), epsilon, del_, S[T].pq.queue[0][0], toc(time_started)
            ))
    
        # Write the Output
        currNode_pt = S[T].pq.queue[0][1]   # Get Optimal Node
        backoutPath(output, robot, currNode_pt) # Backout Optimal Path and action indices.
        output.outputCovariance = currNode_pt.Sigma
        output.target_path = y_T
        if log:
            logPaths(output, robot, S[T], T) # Log all output paths only
    

    def PlanFVI(self, robot, T, debug, log):
    output = PlannerOutput()
    x0 = robot.getState()
    start_idx = robot.env.state_to_idx(x0)

    # Begin Algorithm Clock
    time_started = datetime.now()

    y_T = robot.tmm.predictTargetState(T)  # Predict target trajectory for t = 1,...T-1
    Sigma = robot.tmm.getCovarianceMatrix()  # Get initial Sigma at t=0.

    # Initialize S0
    S0 = InfoStateSpace()
    # Initialize Start Node
    currNode_pt = InfoState(start_idx, x0, Sigma, 0, True)

    currNode_pt.g = cost_function.computeCost(currNode_pt.Sigma)
    currNode_pt.is_open = True
    currNode_pt.state_list.append((currNode_pt.g, currNode_pt))

    # Insert type into the Map
    currNode_pt.heapkey = S0.pq.push((currNode_pt.g, currNode_pt))
    S0.hm[start_idx] = currNode_pt

    # Initialize the list of state spaces
    S = [S0]
    for t in range(1, T+1):
        S_t = InfoStateSpace()
        S.append(S_t)

    # Main Planning Call.
    ImprovePath(S, robot, y_T, 0, 0, float('inf'), tic(), False)

    if S[T].pq.size() == 0:
        print("No feasible path in desired Horizon.")
        S[T] = S[1]

    # Compute final size of tree...
    if debug:
        count = 0
        true_count = 0
        for it in S[T].pq:
            count += 1
            true_count += len(it.second.state_list)
        print(f"FVI Final parameters: T= {T} tree size = {S[T].pq.size()} " 
              f"optimal cost = {S[T].pq.top().second.g} time(total) = {toc(time_started)}")

    # Write the Output
    currNode_pt = S[T].pq.top().second  # Get Optimal Node
    backoutPath(output, robot, currNode_pt)  # Backout Optimal Path and action indices.
    output.outputCovariance = currNode_pt.Sigma
    output.target_path = y_T
    if log:
        logPaths(output, robot, S[T], T)  # Log all output paths only if requested.

    return output
