import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from plotting import *

data = sio.loadmat('data/results_smallVelocity.mat')  # Load the desired MAT File

results = data['results']
GroundTruth = results['GroundTruth'][0, 0]
InitialBelief = results['InitialBelief'][0, 0]
Grid = results['Grid'][0, 0]
CostOfBestPath = results['CostOfBestPath']
Horizon = results['Horizon'][0, 0][0, 0]
RobotTheta = results['RobotTheta'][0, 0]
RobotPositions = results['RobotPositions'][0, 0]
ControlInputs = results['ControlInputs']  # Unused.
Covariance = results['Cov'][0, 0]
SensingRange = results['SensingRange'][0, 0]

# Generate Map
mapmin = np.array([0, 0])
mapmax = np.array([10, 10])
mapres = np.array([0.2, 0.2])
mapsize = np.ceil((mapmax - mapmin) / mapres).astype(np.uint16)
cmap = np.ones((mapsize[0], mapsize[1]))

# Fill in Map with Obstacles
cell = np.round((Grid - mapmin) / mapres).astype(np.uint32)
cmap[cell[:, 0], cell[:, 1]] = 0

plotter = InfoPlotter(mapmin, mapmax, cmap)
plotter.draw_env()

# Simulation Parameters
num_targets = int(np.shape(InitialBelief)[0] / 2)
num_robots = int(np.shape(RobotPositions)[1] / 2)
states = np.split(InitialBelief, num_targets)
tau = 0.1

# plotter.save_cmap()  # Save the CMap to a file for usage by others

for t in range(0, Horizon):
    # Reset the plotter on each iteration
    plotter.clear_plot()
    plotter.draw_env()

    robot_pos_list = [RobotPositions[t][2 * i:2 * (i + 1)] for i in range(0, num_robots)]
    robot_th_list = [RobotTheta[t][i:i + 1] for i in range(0, num_robots)]

    for robot_pos, robot_th in zip(robot_pos_list, robot_th_list):
        robot_state = np.concatenate((robot_pos, robot_th))
        plotter.draw_robot(robot_state, size=0.1)
        plotter.draw_fov(robot_state, SensingRange, 360)

    cov_list = [Covariance[0][t][2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] for i in range(0, num_targets)]
    for state, cov in zip(states, cov_list):  # Plot all targets position and covariance
        plotter.draw_cov(state, cov)
    plt.pause(tau)
    # End Loop

plt.pause(1e3)
plt.show()
