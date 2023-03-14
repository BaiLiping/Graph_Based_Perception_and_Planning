from typing import Dict, List, Tuple
import numpy as np
from target_model import Target, TargetModel


class InfoTarget(Target):
    def __init__(self, target: Target, covariance: np.ndarray):
        super().__init__(target.ID, target.y_dim)
        self.covariance = covariance

    def predict_state(self, T: int) -> np.ndarray:
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return super().get_state()

    def get_covariance(self) -> np.ndarray:
        return self.covariance

    def update_belief(self, mean: np.ndarray, cov: np.ndarray, map_: np.ndarray, cmap: List[str]) -> None:
        raise NotImplementedError



class InfoTargetModel(TargetModel):
    def __init__(self, map_: np.ndarray, cmap: List[str]):
        super().__init__(map_, cmap)

    def add_target(self, ID: int, info_target: InfoTarget) -> bool:
        result = self.add_shared_target(ID, info_target)
        return result

    def get_covariance_matrix(self) -> np.ndarray:
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets.values():
            info_target = target
            result[index:index+info_target.y_dim, index:index+info_target.y_dim] = info_target.get_covariance()
            index += info_target.y_dim
        return result

    def update_belief(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        index = 0
        for target in self.targets.values():
            info_target = target
            info_target.update_belief(mean[index:index+info_target.y_dim], covariance[index:index+info_target.y_dim, index:index+info_target.y_dim], self.map_, self.cmap)
            index += info_target.y_dim

    def get_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        A = self.get_system_matrix()
        W = self.get_noise_matrix()
        return A, W

    def predict_target_state(self, T: int) -> List[np.ndarray]:
        result = []
        target_state = self.get_target_state()
        A = self.get_system_matrix()
        for t in range(T+1):
            result.append(target_state)
            target_state = A @ target_state
        return result

    def get_target_by_id(self, ID: int) -> InfoTarget:
        return self.targets.get(ID)
