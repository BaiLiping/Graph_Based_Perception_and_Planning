# Written by Heejin Jeong (heejinj@seas.upenn.edu)
# This code is to generate baseline results in order to compare with RL algorithms for the target tracking problems.
# Please be sure to set the same parameter values.
# (As of Jan 25th, 2019, The observation noise matrix in include/igl/sensing/observation_models/range_bearing_sensing.h 
# is diffrent from the one used in RL. 
# Please see https://github.com/coco66/RL.git for further information of RL. To access, email to heejinj@seas.upenn.edu.)

import sys
sys.path.append('lib') # IG Library
sys.path.append('script/python') # Plotting Features
import pyInfoGathering as IG
from plotting import *
import pdb
import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse
import pickle
import os, copy
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type = int, default=0)
parser.add_argument('--display', help='display', type = int, default=0)
parser.add_argument('--nb_controls', help='number of controls', type=int, default=2)
parser.add_argument('--nb_traj_steps', help='number of steps per traj', type=int, default=100)
parser.add_argument('--velocity', help='if target has non zero velocity', type=int, default=1)
parser.add_argument('--nb_targets', help='number of targets', type=int, default=1)
parser.add_argument('--target_init_dist', help='initial distance from a robot to a target', type=float, default=10.0)
parser.add_argument('--target_init_vel', help='initial velocity of each component yx=yy', type=float, default=0.0)
parser.add_argument('--nb_robots', help='number of robots', type=int, default=1)
parser.add_argument('--init_sd', help='init_sigma', type=float, default=30.0)
parser.add_argument('--same_noise',help='boolean, W=W_belief', type=int, default=1)
parser.add_argument('--log_dir', help='log directory', type=str, default='.')
parser.add_argument('--repeat', help='repeat with different seed', type=int, default = 1)
parser.add_argument('--map_name', help='map name', type=str, default='emptyMap')
args = parser.parse_args()

MARGIN = 3.0 # Minimum distance between a target and a robot
"""
def get_reward(cov, boundary_penalty=0.0, observed=False):
    done = False
    if not observed :
        reward = -boundary_penalty
    else:
        logdetcov = np.log(LA.det(cov))
        reward = max(-10.0, -0.1*logdetcov - boundary_penalty) #- 0.1*(np.sqrt(pos_dist)-4.0)  #np.log(np.linalg.det(self.target_cov_prev)) - logdetcov 
        reward = 1.0 + max(0.0, reward)
        
    return reward
"""
def get_reward(cov):
    return -np.log(LA.det(cov))

def get_isboundary(pos):
    DIM = 50.0
    return (pos[0] <= -DIM) or (pos[0] >= DIM) or (pos[1] <= -DIM) or (pos[1] > DIM)

def se2_to_cell(pos, mapdim, mapmin, res):
    # pos : xy location [m], 2D
    # OUTPUT : cell - (r,c)
    r = (mapdim[1]-1)-(np.round((pos[1] - mapmin[1])/res[1]).astype(np.int16)-1)
    c = np.round((pos[0] - mapmin[0])/res[0]).astype(np.int16)-1
    return [r, c]

def cell_to_se2(cell_idx, mapdim, mapmin, res): 
    # cell_idx: (r, c)
    cell_x_idx = cell_idx[1]
    cell_y_idx = (mapdim[1]-1)-cell_idx[0]
    return np.array([cell_x_idx, cell_y_idx]) + 1 - 0.5 * res + mapmin

def is_collision(map_arr, pos, mapdim, mapmin, mapmax, res, margin=0.5):
    n = np.ceil(margin/np.array(res)).astype(np.int16)
    if 'empty' in args.map_name:
        return (pos[0] < mapmin[0]+margin) or (pos[0] > mapmax[0]-margin) or (pos[1] < mapmin[1]+margin) or (pos[1] > mapmax[1]-margin)
  
    cell = np.minimum([mapdim[0]-1, mapdim[1]-1] , se2_to_cell(pos, mapdim, mapmin, res))
    for r_add in np.arange(-n[1],n[1],1):
        for c_add in np.arange(-n[0],n[0],1):
            if map_arr[np.clip(cell[0]+r_add,0,mapdim[0]-1).astype(np.int16), np.clip(cell[1]+c_add, 0, mapdim[1]-1).astype(np.int16)] == 1:
                return True
    return False

if __name__== "__main__":
    seed = args.seed

    # Planner Params
    T = 6
    delta = 1.5
    eps = 0
    arvi_time = 1
    range_limit = np.infty
    debug = 1
    # Simulation Params
    tau = 0.5
    n_controls = args.nb_controls
    Tmax = args.nb_traj_steps
    SensingRange = 10.0
    fov = 120
    r_sigma = .15
    b_sigma = .001
    # Assign Map parameters
    cmap_file = 'data/maps/emptySmall/'+args.map_name+'.cfg'
    cmap_yaml = yaml.load(open('data/maps/emptySmall/'+args.map_name+'_RL.yaml', 'r'))
    mapmin = cmap_yaml['mapmin']
    mapmax = cmap_yaml['mapmax']
    mapres = cmap_yaml['mapres']
    map_arr = np.loadtxt(cmap_file).astype(np.int8)
    mapdim = cmap_yaml['mapdim']#map_arr.shape
    map_nd = IG.map_nd(mapmin, mapmax, mapres)
    total_map_size = map_nd.size()[0] * map_nd.size()[1]

    print('map_arr size: ', mapdim, '= ', mapdim[0] * mapdim[1])
    print('map cells: ', total_map_size)

    if 'empty' in args.map_name:
        cmap_data = list(map(str, [0] * total_map_size))
        map_arr = None
    else:
        cmap_data = list(map(str, np.squeeze(map_arr.reshape(-1, 1)).tolist()))
    se2_env = IG.SE2Environment(map_nd, cmap_data, 'data/mprim/mprim_SE2_RL.yaml')
    agent_init_pos = np.array([10.0, 10.0, 0.0])
    if args.velocity:
        A = np.identity(4)
        A[0:2, 2:4] = tau * np.identity(2)
        q = 0.01 # Noise Diffusion Parameter
        W = np.zeros((4,4))
        W[0:2, 0:2] = tau ** 3 / 3 * np.identity(2)
        W[0:2, 2:4] = tau ** 2 / 2 * np.identity(2)
        W[2:4, 0:2] = tau ** 2 / 2 * np.identity(2)
        W[2:4, 2:4] = tau * np.identity(2)
        W = q * W
        Sigma = args.init_sd * np.identity(4)
        if args.same_noise:
            W_belief = W
        else:
            W_belief = 0.02 * np.concatenate((
                        np.concatenate((tau**2/2*np.eye(2), tau/2*np.eye(2)), axis=1),
                        np.concatenate((tau/2*np.eye(2), tau*np.eye(2)),axis=1) ))
    else:
        A = np.identity(2)
        q = 0.01
        W = q * np.identity(2)
        Sigma = args.init_sd * np.identity(2)
    # Construct Sensor
    rb_sensor = IG.RangeBearingSensor(SensingRange, fov, r_sigma, b_sigma, map_nd, cmap_data)

    results = {'init_target': [], 'init_b_target': [], 'rewards':[]}
    # Assign Target Parameters
    for rep in range(args.repeat):
        np.random.seed(seed)
        world = IG.target_model(map_nd, cmap_data)
        info_tmm = IG.info_target_model(map_nd, cmap_data)
        if args.velocity:
            Y = []
            Y_beliefs = []
            for i in range(args.nb_targets):
                is_col = True
                t_init_vel = args.target_init_vel*np.ones((2,))
                while(is_col):
                    rand_ang = np.random.rand()*2*np.pi - np.pi 
                    t_r = np.random.rand()*(args.target_init_dist-MARGIN) + MARGIN
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + agent_init_pos[:2]
                    if np.sqrt(np.sum((t_init - agent_init_pos[:2])**2)) < MARGIN: # robot init pose
                        t_init[0] -= MARGIN
                        t_init[1] -= MARGIN
                    is_col = is_collision(map_arr, t_init, mapdim, mapmin, mapmax, mapres)
                y0 = np.concatenate((t_init, t_init_vel))
                Y.append(y0)
                Y_beliefs.append(np.concatenate((t_init + 10*(np.random.rand(2)-0.5), np.zeros(2))))
            # Add Targets to world
            [world.addTarget(i, IG.DoubleInt2D(i, Y[i][:2], Y[i][2:4], 1.0, tau, q)) for i in range(len(Y))]  # Add All targets to world
            # Add Targets to robot Belief model            
            [info_tmm.addTarget(i, IG.DoubleInt2DBelief(
                    IG.DoubleInt2D(i, Y_beliefs[i][:2], Y_beliefs[i][2:4], 1.0, tau, q), Sigma)) for i in range(len(Y))]  # Add All targets to belief
        else:
            Y = []
            for i in range(args.nb_targets):
                t_init_vel = args.target_init_vel*np.ones((2,))
                rand_ang = np.random.rand()*2*np.pi - np.pi 
                t_r = np.random.rand()*(args.target_init_dist-MARGIN) + MARGIN
                t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)])
                if np.sqrt(np.sum((t_init - 0.0)**2)) < MARGIN: # robot init pose
                    t_init[0] -= MARGIN
                    t_init[1] -= MARGIN
                Y.append(t_init)
            Y_beliefs = Y
            # Add Targets to world
            [world.addTarget(i, IG.Static2D(i, Y[i], q)) for i in range(len(Y))]  # Add All targets to world
            # Add Targets to robot Belief model
            [info_tmm.addTarget(i, IG.Static2DBelief(IG.Static2D(i, Y[i], q), Sigma)) for i in range(len(Y))]
        
        # Initialize robot
        robots = []
        for i in range(args.nb_robots):
            x0 = IG.SE3Pose(agent_init_pos, np.array([0, 0, 0, 1]))
            robots.append(IG.Robot(x0, se2_env, info_tmm, rb_sensor))

        # Initialize Planner
        planner = IG.InfoPlanner()

        # Setup Plotting Environment
        if args.display:
            map_env = robots[0].env.map
            # np.array(robots[0].env.cmap()).reshape(map_env.size()[0], -1).astype(np.uint16)
            plotter = InfoPlotter(map_env.min(), map_env.max(), map_arr)

            #plotter.draw_env()
            #plotter.plot_state(robots, robot_size=2, SensingRange=SensingRange, fov=fov, targets=world.targets)

        # Save Planner Output
        plannerOutputs = [0] * len(robots)
        rewards = []
        print("Experiment #%d"%rep)
        print("Init Target: ", Y[0])
        print("Init Belief: ", Y_beliefs[0], Sigma)
        # Main Loop
        for t in range(Tmax):
            print("Robot position:", robots[0].getState().position, robots[0].getState().getYaw())
            # Reset the plotter on each iteration
            if args.display:
                plotter.clear_plot()
                plotter.draw_env()
                plotter.plot_state(robots, robot_size=2, SensingRange=SensingRange, fov=fov, targets=world.targets)
                if t > 1:
                    plotter.draw_target_path(np.array([plannerOutputs[i].target_path]), clr='g', zorder=2)
                    plotter.draw_paths([plannerOutputs[i].path], clr='b', zorder=2)  # Draw Optimal Path in Blue

            plt.pause(0.5)

            # Plan (Every n_controls steps)
            if t % n_controls == 0:
                for i in range(0, len(robots)):
                    plannerOutputs[i] = planner.planRVI(robots[i], T, delta, eps, debug, 0)

            # Actuate
            for i in range(len(robots)):
                if np.sqrt((robots[i].getState().position[0]-world.getTargetState()[0])**2 \
                    + (robots[i].getState().position[1]-world.getTargetState()[1])**2) < MARGIN:
                    # only bearing control (Temp)
                    copied_act_idx = copy.deepcopy(plannerOutputs[i].action_idx)
                    copied_act_idx[-1] = 9 + plannerOutputs[i].action_idx[-1]%3
                else:
                    copied_act_idx = plannerOutputs[i].action_idx
                # Apply Control
                robots[i].applyControl(copied_act_idx, 1)
                # Pop off last Action manually (UGLY)
                plannerOutputs[i].action_idx = plannerOutputs[i].action_idx[:-1]

            # Update World
            world.forwardSimulate(1)

            # Sense, Filter, Update belief
            for i in range(len(robots)):
                measurements = robots[i].sensor.senseMultiple(robots[i].getState(), world)
                GaussianBelief = IG.MultiTargetFilter(measurements, robots[i], debug=False)
                robots[i].tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)
                print("Observed?",  measurements[0].validity)
                #isBoundary = float(get_isboundary(robots[0].getState().position))
                #rewards.append(get_reward(GaussianBelief.cov, isBoundary, measurements[0].validity))
                #print(GaussianBelief.cov)
                rewards.append(get_reward(GaussianBelief.cov))
                

        # End Loop
        # Output Results
        for i in range(len(robots)):
            print("Final Estimate of robot_", i, " mean=", robots[i].tmm.getTargetState(),"\ncovariance=",
                  robots[i].tmm.getCovarianceMatrix())
            print("Final Pose of robot_", i, " is", robots[i].getState().position, "\n")
        print("Ground Truth State: ", world.getTargetState())
        print('Total reward (episode reward): %.2f'%(np.sum(rewards)))

        results['init_target'].append(Y)
        results['init_b_target'].append(Y_beliefs)
        results['rewards'].append(np.array(rewards))

        seed += 1

    if args.repeat == 1:
        f = open(os.path.join(args.log_dir,'result_%d'%args.seed + '.pkl'),'wb')
    else:
        f = open(os.path.join(args.log_dir,'result_batch_%d'%args.repeat + '.pkl'),'wb')

    pickle.dump(results, f)

