import numpy as np

class SE3Pose:
    def __init__(self, position=None, orientation=None, quaternion=None, se2_pose=None):
        if position is None and orientation is None and quaternion is None and se2_pose is None:
            self.position = np.array([0, 0, 0])
            self.orientation = np.identity(3)
        elif position is not None and orientation is not None:
            self.position = position
            self.orientation = orientation
            if abs(np.linalg.det(orientation) - 1.0) > 1e-3:
                raise ValueError("Invalid Orientation. Not in SO(3)!")
        elif position is not None and quaternion is not None:
            self.position = position
            self.orientation = self.quat2rot(quaternion)
        elif se2_pose is not None:
            self.position = np.array([se2_pose[0], se2_pose[1], 0.0])
            self.orientation = self.rotz(se2_pose[2])
        else:
            raise ValueError("Invalid input arguments")

    @staticmethod
    def quat2rot(quaternion):
        qx, qy, qz, qw = quaternion
        return np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
        ])

    @staticmethod
    def rotz(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def get_yaw(self):
        return np.arctan2(self.orientation[1, 0], self.orientation[0, 0])

    def get_pitch(self):
        return np.arctan2(-self.orientation[2, 0], np.hypot(self.orientation[2, 1], self.orientation[2, 2]))

    def get_roll(self):
        return np.arctan2(self.orientation[2, 1], self.orientation[2, 2])

    def get_se2(self):
        return np.array([self.position[0], self.position[1], self.get_yaw()])

    # TODO: Implement rotation matrix to Quaternion conversion.
    # def get_quaternion(self):
    #     pass
