from typing import List, Tuple, TypeVar, Callable, Optional
import numpy as np

State = TypeVar("State")
MPrim = Tuple[np.ndarray, np.ndarray]


class Environment:
    def __init__(
        self,
        map_in: "MapND",
        cmap: List[int],
        goal: Optional[State] = None,
    ):
        self.cmap_ = cmap
        self.map = map_in
        self.goal_pose = goal
        self.mprim_: List[MPrim] = []

    def get_succ(
        self,
        curr: State,
        succ: List[State],
        succ_cost: List[float],
        action_idx: List[int],
    ) -> None:
        raise NotImplementedError("get_succ method must be implemented in a subclass")

    def forward_action(
        self, curr: State, action_id: int, next_micro: List[State]
    ) -> None:
        raise NotImplementedError("forward_action method must be implemented in a subclass")

    def state_to_idx(self, state: State) -> int:
        raise NotImplementedError("state_to_idx method must be implemented in a subclass")

    def state_to_cell(self, state: State) -> List[int]:
        raise NotImplementedError("state_to_cell method must be implemented in a subclass")

    def state_to_SE2(self, state: State) -> np.ndarray:
        raise NotImplementedError("state_to_SE2 method must be implemented in a subclass")

    def compute_state_metric(self, state1: State, state2: State) -> float:
        raise NotImplementedError("compute_state_metric method must be implemented in a subclass")

    def get_cost_map(self) -> List[int]:
        return self.cmap_
