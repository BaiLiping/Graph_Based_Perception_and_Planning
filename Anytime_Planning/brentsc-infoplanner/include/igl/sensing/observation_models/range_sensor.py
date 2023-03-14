import numpy as np
from math import pi, atan2
from typing import List, Tuple
from sensor import sensor

class RangeSensor(Sensor):
    def __init__(self, min_range, max_range, min_hang, max_hang, min_vang, max_vang, r_sigma, map, cmap):
        super().__init__(sensor_dim=1)
        self.min_range = min_range
        self.max_range = max_range
        self.min_hang = pi / 180 * min_hang
        self.max_hang = pi / 180 * max_hang
        self.min_vang = pi / 180 * min_vang
        self.max_vang = pi / 180 * max_vang
        self.r_sigma = r_sigma
        self.map = map
        self.cmap = cmap

    def observation_model(self, x, y):
        y_ = np.array([y[0], y[1], 0])
        z = np.array([self.compute_range(x.position, y_)])
        return z

    def sense(self, x, target):
        y = target.get_position()
        z = self.observation_model(x, y)
        z[0] += np.random.normal(0, self.r_sigma * self.r_sigma)
        measurement = Measurement(z, target.ID, self.is_valid(x.orientation, x.position, y))
        return measurement

    def get_jacobian(self, H, V, x, y, check_valid):
        V.fill(0)  # Reset Jacobian Matrices
        H.fill(0)
        y_ = np.array([y[0], y[1], 0])
        z = self.observation_model(x, y_)

        if not check_valid or self.is_valid(x.orientation, x.position, y_):
            H[0, 0] = y[0] - x.position[0]
            H[0, 1] = y[1] - x.position[1]
            H /= (0.001 + z[0])

        V[0, 0] = self.r_sigma * self.r_sigma

    # Other methods (including private ones) should be added here

    def compute_range(self, x, y):
        return np.linalg.norm(x - y)

    def is_valid(self, R, p, y):
        y_laser_frame = R.T @ (y - p)
        azimuth = atan2(y_laser_frame[1], y_laser_frame[0])

        if azimuth <= self.min_hang or azimuth >= self.max_hang:
            return False

        d = np.linalg.norm(p - y)
        if d <= self.min_range or d >= self.max_range:
            return False

        collision = self.ray_trace(p, y)
        return not collision

    def max_sensor_matrix(self, x, y, tmm, T):
        M = np.zeros((y.shape[0], y.shape[0]))
        index = 0

        for target_id, target in tmm.targets.items():
            y_dim = target.y_dim
            z = self.observation_model(x, y[index * y_dim : (index + 1) * y_dim])

            if z[0] < 1.5 * T + self.max_range:
                M[
                    index * y_dim : (index + 1) * y_dim,
                    index * y_dim : (index + 1) * y_dim,
                ] = (1.0 / (self.r_sigma * self.r_sigma)) * np.identity(y_dim)

            index += 1

        return M