import sys

sys.path.append('lib')  # IG Library
sys.path.append('script/python')  # Plotting Features
import pyInfoGathering as IG
from plotting import *
import matplotlib.pyplot as plt
import pdb
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--display', help='display', type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    # Load Parameters from YAML
    params = IG.Parameters('data/init_info_planner_ARVI.yaml')
    robots = params.GetRobots()
    world = params.GetTMM()
    planner = params.GetPlanner()

    # Setup Plotting Environment
    if args.display:
        map = robots[0].env.map
        # cmap = np.array(robots[0].env.cmap()).reshape(map.size()[0], -1).astype(np.uint16)
        plotter = InfoPlotter(map.min(), map.max(), None)

    # Save Planner Output
    plannerOutputs = [0] * len(robots)

    # Main Loop
    for t in range(0, params.Tmax):
        # Sense, Filter, Update belief
        for i in range(0, len(robots)):
            measurements = robots[i].sensor.senseMultiple(robots[i].getState(), world)
            GaussianBelief = IG.MultiTargetFilter(measurements, robots[i], debug=False)
            robots[i].tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

        # Plan (Every n_controls steps)
        if t % params.n_controls == 0:
            for i in range(0, len(robots)):
                plannerOutputs[i] = planner.planARVI(robots[i], 12, 3, np.infty, 1, 1, 1)

        # Actuate
        for i in range(0, len(robots)):
            # Apply Control
            robots[i].applyControl(plannerOutputs[i].action_idx, 1)
            # Pop off last Action (UGLY)
            plannerOutputs[i].action_idx = plannerOutputs[i].action_idx[:-1]

        # Update World
        world.forwardSimulate(1)

        # Draw State
        if args.display:
            plotter.plot_state(robots, robot_size=0.3, SensingRange=8, fov=360, targets=world.targets)
            plt.pause(0.1)
    # End Loop

    # Output Results
    for i in range(0, len(robots)):
        print("Final Estimate of robot_", i, " mean=", robots[i].tmm.getTargetState(), "\ncovariance=",
              robots[i].tmm.getCovarianceMatrix())
        print("Final Pose of robot_", i, " is", robots[i].getState().position, "\n")
    print("Ground Truth State: ", world.getTargetState())
