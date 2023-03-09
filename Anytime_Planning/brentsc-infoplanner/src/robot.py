from typing import List
import igl.robot
import igl.utils.utils_nx
import igl.se3_pose


class Robot:
    def __init__(self, env, x):
        self.env = env
        self.x = x

    def applyControl(self, action_idx: List[int], n_controls: int):
        if not action_idx:
            print("WARNING: No control inputs available for robot to apply.")
            return

        next_micro = []
        i = 0
        for it in reversed(action_idx):
            if i >= n_controls:
                break
            self.env.forward_action(self.getState(), action_idx[-1], next_micro)
            self.x = next_micro[-1]  # Move the robot's position.
            action_idx.pop()

    def getState(self):
        return self.x

    def setState(self, s):
        self.x = s
