from typing import List, TypeVar, Generic

State = TypeVar('State')

class Robot(Generic[State]):
    def __init__(self, env: Any, x: State, tmm: Any, sensor: Any) -> None:
        self.env = env
        self.x = x
        self.tmm = tmm
        self.sensor = sensor

    def apply_control(self, action_idx: List[int], n_controls: int) -> None:
        # If the robot has no actions, give a warning.
        if len(action_idx) == 0:
            print("WARNING: No control inputs available for robot to apply.")

        next_micro = []
        for i in range(min(n_controls, len(action_idx))):
            self.env.forward_action(self.get_state(), action_idx.pop(), next_micro)
            self.x = next_micro[-1]  # Move the robot's position.

    def get_state(self) -> State:
        return self.x

    def set_state(self, s: State) -> None:
        self.x = s
