import yaml
import numpy as np
from include.igl.params import Parameters
from include.igl.env.env_se2 import EnvSE2
from include.igl.planning.infoplanner import InfoPlanner
from include.igl.mapping.map_nx import MapNX
from include.igl.utils.utils_nx import tic, toc
from include.igl.robot import Robot
from include.igl.estimation.kalman_filter import KalmanFilter
from include.igl.estimation.multi_target_filter import MultiTargetFilter
from include.igl.se3_pose import SE3Pose

if __name__ == '__main__':
    # Replace 'config_file' with the path to your configuration file
    config_file = 'config_file.yaml'
    params = Parameters(config_file)

    robots = params.get_robots()
    planner = params.get_planner()
    world = params.get_tmm()

    fixed_traj = [[] for _ in range(len(robots))]
    planner_outputs = [None] * len(robots)

    t1 = tic()

    for t in range(params.Tmax):
        print(f"Timestep: {t}")

        # Sense and Filter
        for i, robot in enumerate(robots):
            measurements = robot.sensor.sense_multiple(robot.get_state(), world)
            output = MultiTargetFilter.multi_target_kf(measurements, robot)
            robot.tmm.update_belief(output.mean, output.cov)

        # PlanARVI (Every n_controls steps)
        if t % params.n_controls == 0:
            print("Starting InfoPlanner...")

            # Reset planning variables
            for i in range(len(robots)):
                fixed_traj[i].clear()

            for i, robot in enumerate(robots):
                planner_outputs[i] = planner.plan_arvi(robot, 12)

                # Record fixed trajectories for Coordinate Descent
                for state in planner_outputs[i].path:
                    point = robot.env.state_to_se2(state)
                    fixed_traj[i].append(point)

            print(f"Computation done in {toc(t1)} sec!")

            # TODO: Update the collision map to avoid robots crossing trajectories

        # Actuate
        for i, robot in enumerate(robots):
            robot.apply_control(planner_outputs[i].action_idx, 1)

        # Environment Changes
        world.forward_simulate(1)

    print(f"Final ground truth is: {world.get_target_state()}")

    for i, robot in enumerate(robots):
        print(f"Final Estimate_{i} is: {robot.tmm.get_target_state()}")
        print(f"Final cov is: {robot.tmm.get_covariance_matrix()}")
        print(f"Final pose_{i} is: {robot.get_state().position}")
