import sys

sys.path.append('lib')  # IG Library
import pyInfoGathering as IGL
import numpy as np

'''
Fixed Target Locations
'''
# y0 = np.array([3.5, 6.8])
# y1 = np.array([3.5, 2.2])
# y2 = np.array([6.5, 2.2])
# y3 = np.array([7.7, 6.8])
# y4 = np.array([3.5, 4])

y0 = np.array([3.5, 6.8])
y1 = np.array([3.5, 2.2])
y2 = np.array([6.5, 2.2])
y3 = np.array([6.5, 6.8])
y4 = np.array([3.5, 4])
y5 = np.array([6.5, 4.8])

Y = [y0, y1, y2, y3, y4, y5]


class Policy:
    """
    Policy class contains several simple policy implementations for SE2 moving targets.
    """

    def zero_policy(state):
        """
        Policy returns a zero control input.
        :return: Control input of v=0, w=0.
        """
        return np.array([0, 0])

    def linear_policy(speed):
        """
        Returns a linear policy with the requested speed.
        :return: Control input with v=speed, w=0.
        """
        return lambda state: np.array([speed, 0])


class Configure(object):
    """
    Configure class manages the construction of targets.
    """

    def __init__(self, map_nd, cmap):
        """
        Initialize the Configuration Object.
        :param map_nd: The map for boundary checking.
        :param cmap: The costmap for collision checking.
        """
        self.map_nd = map_nd
        self.cmap = cmap

    def setup_integrator_targets(self, n_targets=1, max_vel=1.0, tau=0.5, q=0.0):
        """
        Setup Ground Truth Integrator Moving Targets.
        :param n_targets: The number of targets to add.
        :param max_vel: The maximum velocity targets can attain.
        :param tau: The time discretization.
        :param q: The Noise diffusion parameter for the simulation.
        :return: The world model containing the requested targets.
        """
        world_model = IGL.target_model(self.map_nd, self.cmap)
        vel = np.array([0, 0])
        # Add All targets to belief
        [world_model.addTarget(i, IGL.DoubleInt2D(i, Y[i], vel, tau, max_vel, q)) for i in range(0, n_targets)]
        return world_model

    def setup_integrator_belief(self, n_targets=1, max_vel=1.0, tau=0.5, q=0.0, cov_pos=.25, cov_vel=.1):
        """
        Setup Integrator Belief Model for Moving Targets.
        :param n_targets: The number of targets to add to the belief.
        :param max_vel: The maximum velocity target beliefs can attain.
        :param tau: The time discretization.
        :param q: The Noise diffusion parameter for the model.
        :param cov_pos: The initial covariance in position.
        :param cov_vel: The initial covariance in velocity.
        :return: The belief model containing the requested targets.
        """
        belief_model = IGL.info_target_model(self.map_nd, self.cmap)
        sigma = np.identity(4)
        sigma[0:2, 0:2] = cov_pos * np.identity(2)
        sigma[2:4, 2:4] = cov_vel * np.identity(2)
        vel = np.array([0, 0])
        [belief_model.addTarget(i, IGL.DoubleInt2DBelief(
            IGL.DoubleInt2D(i, Y[i], vel, tau, max_vel, q), sigma)) for i in
         range(0, n_targets)]  # Add All targets to belief

        return belief_model

    def setup_static_targets(self, n_targets=1, q=0.0):
        """
        Set up a Ground Truth static target model.
        :param n_targets: The number of targets to add.
        :param q: The noise parameter of the simulation.
        :return: The requested world model.
        """
        world_model = IGL.target_model(self.map_nd, self.cmap)
        # Add All targets to belief
        [world_model.addTarget(i, IGL.Static2D(i, Y[i], q)) for i in range(0, n_targets)]
        return world_model

    def setup_static_belief(self, n_targets=1, q=0.0, cov_pos=0.25):
        """
        Setup a Static Belief target Model.
        :param n_targets: The number of requested targets in the belief.
        :param q: The noise parameter of the belief model.
        :param cov_pos: The initial position covariance.
        :return: The requested belief model.
        """
        belief_model = IGL.info_target_model(self.map_nd, self.cmap)
        # Add All targets to belief
        [belief_model.addTarget(i, IGL.Static2DBelief(IGL.Static2D(i, Y[i], q), cov_pos * np.identity(2))) for i in
         range(0, n_targets)]
        return belief_model

    def setup_se2_targets(self, n_targets=1, policy=Policy.zero_policy, tau=0.5, q=0.0):
        """
        Set up a Ground Truth simulation of SE(2) moving targets.
        :param n_targets: The number of targets requested.
        :param policy: The control policy used for the targets.
        :param tau: The time discretization.
        :param q: The Noise parameter.
        :return: The requested belief model.
        """
        world_model = IGL.target_model(self.map_nd, self.cmap)
        yaw = 0
        controller = IGL.SE2Policy(policy)
        [world_model.addTarget(i, IGL.SE2Target(i, np.array([Y[i][0], Y[i][1], yaw]), policy=controller, tau=tau,
                                                     q=q))
         for i in
         range(0, n_targets)]
        return world_model
