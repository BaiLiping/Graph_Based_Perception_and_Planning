import numpy as np
from se3_pose import SE3Pose
from environment import Environment
from motion_primitive import MotionPrimitive
from dd_motion_model import dd_motion_model
from mapping import mprms_from_yaml, meters2cells, cells2meters
from se3_pose import rotz, restrict_angle
from map_nd import MapND

PI = np.pi

class SE2Environment(Environment[SE3Pose]):

    def __init__(self, map_in: MapND, cmap_ptr: list[str], mprim_yaml: str, goal: SE3Pose = SE3Pose()):
        super().__init__(map_in, cmap_ptr, goal)
        self.goal_coord = goal
        self.max_len_traj_ = 0
        self.samp = 0.5
        self.yaw_discretization_size = 60
        self.yaw_res = 2 * PI / self.yaw_discretization_size
        self.is_3d = len(map_in.res()) == 3
        self.mprim_xd_ = []

        self.mprim_ = mprms_from_yaml(mprim_yaml, dd_motion_model, np.array([0, 0, 0]), self.samp)

        self.init_mprim_xd_()

        for tr in range(len(self.mprim_)):
            if len(self.mprim_[tr].uVec) > self.max_len_traj_:
                self.max_len_traj_ = len(self.mprim_[tr].uVec)

    def init_mprim_xd_(self):
        x, y, q = 0, 0, 0
        xc, yc, qc = 0, 0, 0
        theta_sz = self.yaw_discretization_size
        self.mprim_xd_ = [[] for _ in range(theta_sz)]
        num_prim = len(self.mprim_)
        for k in range(theta_sz):
            self.mprim_xd_[k] = [[] for _ in range(num_prim)]
            ori = cells2meters(k, -np.pi, self.yaw_res)
            for pr in range(num_prim):
                num_seg = len(self.mprim_[pr]["xVecVec"])
                self.mprim_xd_[k][pr] = [[] for _ in range(num_seg)]
                for seg in range(num_seg):
                    num_sta = len(self.mprim_[pr]["xVecVec"][seg])
                    for st in range(num_sta):
                        x, y, q = smart_plus_SE2(self.map.origin[0], self.map.origin[1], ori,
                                                 self.mprim_[pr]["xVecVec"][seg][st][0],
                                                 self.mprim_[pr]["xVecVec"][seg][st][1],
                                                 self.mprim_[pr]["xVecVec"][seg][st][2])
                        xc = meters2cells(x, self.map.min[0], self.map.res[0]) - self.map.origincells[0]
                        yc = meters2cells(y, self.map.min[1], self.map.res[1]) - self.map.origincells[1]
                        qc = meters2cells(q, -np.pi, self.yaw_res)
                        
                        # add only if unique
                        if not self.mprim_xd_[k][pr][seg] or (xc != self.mprim_xd_[k][pr][seg][-1][0] or
                                                              yc != self.mprim_xd_[k][pr][seg][-1][1] or
                                                              qc != self.mprim_xd_[k][pr][seg][-1][2]):
                            self.mprim_xd_[k][pr][seg].append((xc, yc, qc))


    def computeStateMetric(self, state1, state2):
        """
        Compute the distance between two states in the SE2 environment.

        Args:
        state1: tuple (x1, y1, theta1), where x1 and y1 are the Cartesian coordinates
                and theta1 is the orientation of the first state.
        state2: tuple (x2, y2, theta2), where x2 and y2 are the Cartesian coordinates
                and theta2 is the orientation of the second state.

        Returns:
        float: The distance between the two states.
        """
        x1, y1, theta1 = state1
        x2, y2, theta2 = state2

        position_weight = 1.0
        orientation_weight = 0.5

        delta_x = x1 - x2
        delta_y = y1 - y2
        delta_theta = np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))

        position_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
        orientation_distance = np.abs(delta_theta)

        distance = position_weight * position_distance + orientation_weight * orientation_distance

        return distance


    def forward_action(self, state, action):
        """
        Compute the next state after taking an action from the current state in the SE2 environment.

        Args:
        state: tuple (x, y, theta), where x and y are the Cartesian coordinates
               and theta is the orientation of the current state.
        action: tuple (delta_x, delta_y, delta_theta), where delta_x and delta_y are the
                changes in Cartesian coordinates and delta_theta is the change in orientation.

        Returns:
        tuple: The next state (x', y', theta') after taking the action.
        """
        x, y, theta = state
        delta_x, delta_y, delta_theta = action

        # Update position and orientation
        x_new = x + delta_x * np.cos(theta) - delta_y * np.sin(theta)
        y_new = y + delta_x * np.sin(theta) + delta_y * np.cos(theta)
        theta_new = (theta + delta_theta) % (2 * np.pi)

        # Return the new state
        return x_new, y_new, theta_new
