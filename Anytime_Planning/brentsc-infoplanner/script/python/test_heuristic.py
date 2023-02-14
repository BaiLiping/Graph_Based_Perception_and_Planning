import sys

sys.path.append('lib')  # IG Library
sys.path.append('script/python')  # Plotting Features
import pyInfoGathering as IGL
from configure_targets import *

from plotting import *
import pdb
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--display', help='display', type=int, default=0)
args = parser.parse_args()

if __name__ == "__main__":

    # Initialize Simulation Params
    tau = 0.5
    n_controls = 5

    log = 1

    # Initialize Planner
    T = 12
    delta = 3
    eps = np.infty
    arvi_time = 60
    debug = 1

    planner = IGL.InfoPlanner()      # Cost Functions
    planner = IGL.InfoPlanner(IGL.DeterminantCost())
    planner = IGL.InfoPlanner(IGL.TraceCost())

    # Assign Map parameters
    mapmin = [0, 0]
    mapmax = [10, 10]
    mapres = [.01, .01]
    map_nd = IGL.map_nd(mapmin, mapmax, mapres)
    cmap_file = 'data/maps/emptySmall/obstacles_large.cfg'
    cmap_data = ['0'] * map_nd.size()[0] * map_nd.size()[1]
    cmap_data = list(map(str, np.squeeze(np.loadtxt(cmap_file).astype(np.int8).reshape(-1, 1)).tolist()))
    se2_env = IGL.SE2Environment(map_nd, cmap_data, 'data/mprim/mprim_SE2.yaml')

    # Setup Ground Truth Target Simulation
    cfg = Configure(map_nd, cmap_data)
    world = cfg.setup_integrator_targets(n_targets=1, q=0.001)  # Integrator Ground truth Model
    world = cfg.setup_static_targets(n_targets=1, q=0.001)  # Static Ground truth Model
    world = cfg.setup_se2_targets(n_targets=1, policy=Policy.linear_policy(1), q=0)  # SE2 Ground truth Model (linear)
    world = cfg.setup_se2_targets(n_targets=6, policy=Policy.zero_policy, q=0)  # SE2 Ground truth Model (Zero)

    # Setup Belief Model
    info_tmm = cfg.setup_static_belief(n_targets=6, q=0, cov_pos=0.25)  # Static Belief Model
    # info_tmm = cfg.setup_integrator_belief(n_targets=3, q=.01, cov_pos=0.25)  # Integrator Belief Model

    # Construct Sensor
    SensingRange = 1.1
    fov = 360
    r_sigma = .15
    b_sigma = .01

    # sensor = IGL.PositionSensor(0, SensingRange, -fov / 2, fov / 2, -90, 90, r_sigma, map_nd, cmap_data)  # Position Sensor
    # sensor = IGL.RangeBearingSensor(SensingRange, fov, r_sigma, b_sigma, map_nd, cmap_data)  # Range-Bearing sensor
    sensor = IGL.RangeSensor(0, SensingRange, 0, fov, 0, 180, r_sigma, map_nd, cmap_data)  # Range Only sensor.
    # sensor = IGL.BearingSensor(0, SensingRange, 0, fov, b_sigma, map_nd, cmap_data)  # Bearing Only sensor.

    # Initialize robot
    x0 = IGL.SE3Pose(np.array([3, 3, 0]), np.array([0, 0, 0, 1]))
    robot = IGL.Robot(x0, se2_env, info_tmm, sensor)  # , IGL.Robot(x1, se2_env, info_tmm, rb_sensor)]

    # Setup Plotting Environment
    if args.display:
        map = robot.env.map
        cmap = np.array(robot.env.cmap()).reshape(map.size()[0], -1).astype(np.uint16)
        plotter = InfoPlotter(map.min(), map.max(), cmap, title='Anytime RVI')
        plotter_heur = InfoPlotter(map.min(), map.max(), cmap, plot_num=2, title='AStar with Heuristic')
        plotter.draw_env()
        plotter_heur.draw_env()

    # Save Planner Output
    plannerOutputs = [0]
    plannerOutputsHeur = [0]

    # Main Loop
    print('Beginning Planning ')
    # Reset the plotter on each iteration
    if args.display:
        plotter.clear_plot()
        plotter.draw_env()

    # Sense, Filter, Update belief
    measurements = robot.sensor.senseMultiple(robot.getState(), world)
    GaussianBelief = IGL.MultiTargetFilter(measurements, robot, debug=False)
    robot.tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

    # Planning
    plannerOutputsHeur = planner.planAstar(robot, T, 2.5, debug, log)
    print(plannerOutputsHeur.action_idx)
    if T < 6:  # If short horizon use FVI
        plannerOutputs = planner.planFVI(robot, T, debug, log)
    else:
        plannerOutputs = planner.planARVI(robot, T, delta, np.infty, arvi_time, debug, log)
    print(plannerOutputs.action_idx)

    # Draw State
    if args.display:
        plotter.plot_state([robot], robot_size=0.3, SensingRange=SensingRange, fov=fov, targets=world.targets)
        plotter_heur.plot_state([robot], robot_size=0.3, SensingRange=SensingRange, fov=fov, targets=world.targets)

        plotter.draw_paths([plannerOutputs.path], clr='b', zorder=5)  # Draw Optimal Path in Blue
        plotter_heur.draw_paths([plannerOutputsHeur.path], clr='b', zorder=5)  # Draw Optimal Path in Blue


        plotter.draw_target_path(np.array([plannerOutputs.target_path]), clr='r', zorder=2)
        plotter_heur.draw_target_path(np.array([plannerOutputsHeur.target_path]), clr='r', zorder=2)

        if log:
            plotter.draw_paths(plannerOutputs.all_paths, clr='y', zorder=1)  # Draw Remaining Paths in Cyan
            plotter_heur.draw_paths(plannerOutputsHeur.all_paths, clr='y',
                                    zorder=1)  # Draw Remaining Paths in Cyan
            plotter_heur.draw_observed_points(plannerOutputsHeur.observation_points, range=SensingRange, fov=fov,
                                              clr='c')
            plotter_heur.draw_expanded_nodes(plannerOutputsHeur.closed_list)
            plotter.draw_observed_points(plannerOutputs.observation_points, range=SensingRange, fov=fov, clr='c')

        plotter.add_legend()
        plotter_heur.add_legend()
        plt.pause(0.1)

    # Simulate World
    world.forwardSimulate(1)

    # End Loop
    print('Finished Simulation')
    if args.display:
        plt.pause(1000)