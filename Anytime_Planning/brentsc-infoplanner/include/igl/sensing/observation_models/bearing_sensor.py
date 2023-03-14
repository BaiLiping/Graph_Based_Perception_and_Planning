import numpy as np
from scipy.linalg import block_diag


class BearingSensor(Sensor):
    def __init__(self, z_dim: int, R: np.ndarray):
        super().__init__(z_dim)
        self.R = R

    def observation_model(self, x: 'State', y: np.ndarray) -> np.ndarray:
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        bearing = np.arctan2(dy, dx) - x[2]
        return np.array([bearing])

    def sense(self, x: 'State', y: 'Target') -> Measurement:
        z = self.observation_model(x, y.y)
        z_with_noise = z + np.random.multivariate_normal(np.zeros(self.z_dim), self.R)
        valid = 1
        return Measurement(z_with_noise, y.ID, valid)

    def get_jacobian(self, x: 'State', y: np.ndarray, check_valid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        dist_sq = dx ** 2 + dy ** 2
        dist = np.sqrt(dist_sq)

        H = np.zeros((self.z_dim, y.shape[0]))
        H[0, 0] = -dy / dist_sq
        H[0, 1] = dx / dist_sq

        V = self.R

        return H, V

    def sense_multiple(self, x: 'State', tmm: 'TargetModel') -> List[Measurement]:
        output = []
        for _, target in tmm.targets.items():
            output.append(self.sense(x, target))
        return output

    def get_jacobian_multiple(self, x: 'State', tmm: 'TargetModel', y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        V_list = []
        H_list = []
        index = 0
        for _, target in tmm.targets.items():
            y_dim = target.y_dim
            y_predict = y[index * y_dim:(index + 1) * y_dim]
            H_i, V_i = self.get_jacobian(x, y_predict)
            H_list.append(H_i)
            V_list.append(V_i)
            index += 1

        H = block_diag(*H_list)
        V = block_diag(*V_list)

        return H, V
