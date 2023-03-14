import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class Measurement:
    def __init__(self, z: np.ndarray, ID: int, valid: int):
        self.z = z
        self.ID = ID
        self.valid = valid
        self.z_dim = z.shape[0]


class Sensor(ABC):
    def __init__(self, z_dim: int):
        self.z_dim = z_dim

    @abstractmethod
    def observation_model(self, x: 'State', y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sense(self, x: 'State', y: 'Target') -> Measurement:
        pass

    def compute_innovation(self, measurement: np.ndarray, predicted_measurement: np.ndarray) -> np.ndarray:
        return measurement - predicted_measurement

    @abstractmethod
    def get_jacobian(self, x: 'State', y: np.ndarray, check_valid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def max_sensor_matrix(self, x: 'State', y: np.ndarray, tmm: 'TargetModel', T: int) -> np.ndarray:
        return 1e10 * np.identity(y.shape[0])

    def sense_multiple(self, x: 'State', tmm: 'TargetModel') -> List[Measurement]:
        output = []
        for _, target in tmm.targets.items():
            output.append(self.sense(x, target))
        return output

    def get_jacobian_multiple(self, x: 'State', tmm: 'TargetModel', y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        V = np.zeros((y.shape[0], y.shape[0]))
        H = np.zeros((y.shape[0], y.shape[0]))
        index = 0
        for _, target in tmm.targets.items():
            y_dim = target.y_dim
            y_predict = y[index * y_dim:(index + 1) * y_dim]
            H_i, V_i = self.get_jacobian(x, y_predict)
            H[index * self.z_dim:(index + 1) * self.z_dim, index * y_dim:(index + 1) * y_dim] = H_i
            V[index * self.z_dim:(index + 1) * self.z_dim, index * self.z_dim:(index + 1) * self.z_dim] = V_i
            index += 1
        return H, V

