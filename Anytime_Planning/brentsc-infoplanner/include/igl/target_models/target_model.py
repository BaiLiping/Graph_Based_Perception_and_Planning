from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class Target(ABC):
    def __init__(self, ID: int, y_dim: int):
        self.ID = ID
        self.y_dim = y_dim

    @abstractmethod
    def get_position(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_jacobian(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_noise(self) -> np.ndarray:
        pass

    @abstractmethod
    def forward_simulate(self, T: int, map: 'nx.Map', cmap: List[str]) -> None:
        pass


class TargetModel:
    def __init__(self, map: 'nx.Map', cmap: List[str]):
        self.target_dim = 0
        self.targets: Dict[int, Target] = {}
        self.map = map
        self.cmap = cmap

    def add_target(self, ID: int, target: Target) -> bool:
        if ID not in self.targets:
            self.targets[ID] = target
            self.target_dim += target.y_dim
            return True
        return False

    def remove_target(self, ID: int) -> None:
        target = self.targets[ID]
        self.target_dim -= target.y_dim
        del self.targets[ID]

    def get_target_state(self) -> np.ndarray:
        result = np.zeros(self.target_dim)
        index = 0
        for target in self.targets.values():
            result[index:index + target.y_dim] = target.get_state()
            index += target.y_dim
        return result

    def get_system_matrix(self) -> np.ndarray:
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets.values():
            jacobian = target.get_jacobian()
            result[index:index + target.y_dim, index:index + target.y_dim] = jacobian
            index += target.y_dim
        return result

    def get_noise_matrix(self) -> np.ndarray:
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets.values():
            noise = target.get_noise()
            result[index:index + target.y_dim, index:index + target.y_dim] = noise
            index += target.y_dim
        return result

    def forward_simulate(self, T: int) -> None:
        for target in self.targets.values():
            target.forward_simulate(T, self.map, self.cmap)

    def num_targets(self) -> int:
        return len(self.targets)

    def get_target_by_id(self, ID: int) -> Target:
        return self.targets[ID]
