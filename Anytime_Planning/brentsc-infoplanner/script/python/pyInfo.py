import sys

sys.path.append('lib')  # IG Library
sys.path.append('script/python')  # Plotting Features
import pyInfoGathering as IGL
from plotting import *
from configure_targets import *
import pdb

import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--display', help='display', type=int, default=0)
parser.add_argument('--video', help='video', type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":
    # Initialize Simulation Params
    tau = 0.5
    n_controls = 5
    Tmax = 100

    # Assign Map parameters
    mapmin = [0, 0]
    mapmax = [10, 10]
    mapres = [.2, .2]
    map_nd = IGL.map_nd(mapmin, mapmax, mapres)
    cmap_file = 'data/maps/emptySmall/obstacles.cfg'
    cmap_data = list(map(str, np.squeeze(np.loadtxt(cmap_file).astype(np.int8).reshape(-1, 1)).tolist()))
    se2_env = IGL.SE2Environment(map_nd, cmap_data, 'data/mprim/mprim_SE2.yaml')

    # Setup Ground Truth Target Simulation
    cfg = Configure(map_nd, cmap_data)
    world = cfg.setup_integrator_targets(n_targets=1, q=0.001)  # Integrator Ground truth Model
    #world = cfg.setup_static_targets(n_targets=1, q=0.001)  # Static Ground truth Model
    #world = cfg.setup_se2_targets(n_targets=1, policy=Policy.linear_policy(1), q=0)  # SE2 Ground truth Model (linear)
    #world = cfg.setup_se2_targets(n_targets=3, policy=Policy.zero_policy, q=0)  # SE2 Ground truth Model (Zero)

    # Setup Belief Model
    #info_tmm = cfg.setup_static_belief(n_targets=1, q=.01, cov_pos=0.25)  # Static Belief Model
    info_tmm = cfg.setup_integrator_belief(n_targets=1, q=.001, cov_pos=0.25)  # Integrator Belief Model

    # Setup Sensor
    SensingRange = 10
    fov = 120
    r_sigma = .15
    b_sigma = 0.001

    sensor = IGL.RangeBearingSensor(SensingRange, fov, r_sigma, b_sigma, map_nd, cmap_data)  # Range and Bearing
    # sensor = IGL.PositionSensor(0, SensingRange, -fov / 2, fov / 2, -90, 90,
    #                             r_sigma, map_nd, cmap_data)  # Position Sensor
    # sensor = IGL.BearingSensor(0, SensingRange, -fov / 2, fov / 2, b_sigma, map_nd, cmap_data)  # Bearing sensor.
    # sensor = IGL.RangeSensor(0, SensingRange, -fov / 2, fov / 2, 0, 180, r_sigma, map_nd, cmap_data)  # Range sensor.

    # Initialize robot
    x0 = IGL.SE3Pose(np.array([1, 6, 0]), np.array([0, 0, 0, 1]))
    x1 = IGL.SE3Pose(np.array([.2, 6, 0]), np.array([0, 0, 0, 1]))
    robots = [IGL.Robot(x0, se2_env, info_tmm, sensor)]  # , IGL.Robot(x1, se2_env, info_tmm, sensor)]

    # Initialize Planner
    T = 12
    delta = 3
    eps = np.infty
    arvi_time = 1
    range_limit = np.infty
    debug = 1
    planner = IGL.InfoPlanner()

    # Setup Plotting Environment
    if args.display:
        map = robots[0].env.map
        cmap = np.array(robots[0].env.cmap()).reshape(map.size()[0], -1).astype(np.uint16)
        plotter = InfoPlotter(map.min(), map.max(), cmap, video=args.video)
        plotter.draw_env()

    # Save Planner Output
    plannerOutputs = [0] * len(robots)

    # Main Loop
    for t in range(0, Tmax):
        print('Timestep ', t)
        # Reset the plotter on each iteration
        if args.display:
            plotter.clear_plot()
            plotter.draw_env()

        # Sense, Filter, Update belief
        for i in range(0, len(robots)):
            measurements = robots[i].sensor.senseMultiple(robots[i].getState(), world)
            GaussianBelief = IGL.MultiTargetFilter(measurements, robots[i], debug=True)
            robots[i].tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

        # Plan for individual Robots (Every n_controls steps)
        if t % n_controls == 0:
            for i in range(0, len(robots)):
                plannerOutputs[i] = planner.planARVI(robots[i], T, delta, eps, arvi_time, debug, 0)

        # Actuate
        for i in range(0, len(robots)):
            # Apply Control
            robots[i].applyControl(plannerOutputs[i].action_idx, 1)
            # Pop off last Action manually (UGLY)
            plannerOutputs[i].action_idx = plannerOutputs[i].action_idx[:-1]

        # Update World
        world.forwardSimulate(1)

        # Draw State
        if args.display:
            plotter.plot_state(robots, robot_size=0.3, SensingRange=SensingRange, fov=fov, targets=world.targets)
            plotter.draw_target_path(np.array([plannerOutputs[i].target_path]), clr='g', zorder=2)
            plt.pause(0.1)

    # End Loop
    # plt.pause(1000)
    # Output Results
    for i in range(0, len(robots)):
        print("Final Estimate of robot_", i, " mean=", robots[i].tmm.getTargetState(), "\ncovariance=",
              robots[i].tmm.getCovarianceMatrix())
        print("Final Pose of robot_", i, " is", robots[i].getState().position, "\n")
    print("Ground Truth State: ", world.getTargetState())

    if args.video:
        plotter.save_video('5_target_fast', fps=3)
