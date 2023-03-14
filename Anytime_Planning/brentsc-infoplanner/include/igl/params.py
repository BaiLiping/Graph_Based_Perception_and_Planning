import yaml
import numpy as np
from typing import List, Tuple
from robot import Robot, SE3Pose
from environment import Environment
import numpy as np
import yaml
from SE3Pose import SE3Pose
from info_planner import InfoPlanner
from info_target_model import InfoTargetModel
from target_model import TargetModel
from range_bearing_sensor import RangeBearingSensor
from se2_environment import SE2Environment
class Parameters:
    def __init__(self, yaml_file: str):
        with open(yaml_file, "r") as f:
            node = yaml.safe_load(f)

        self.plannerConfig = node["plannerConfig"]
        self.targetConfig = node["targetConfig"]
        self.samp = node["samp"]
        self.n_controls = node["n_controls"]
        self.Tmax = node["Tmax"]
        self.log = node["log"]
        self.debug = node["debug"]
        self.outputPath = node["outputpath"]

        # Build Simulation Actors
        self.robots = self.build_robots(yaml_file)
        self.p = self.build_planner(self.plannerConfig)
        self.world = self.build_TMM(self.targetConfig)

    def get_robots(self) -> List[Robot[SE3Pose]]:
        return self.robots

    def get_planner(self) -> 'InfoPlanner[SE3Pose]':  # Replace with the correct type once implemented.
        return self.p

    def get_TMM(self) -> 'TargetModel':  # Replace with the correct type once implemented.
        return self.world

    def __init_subclass__(cls):
        """Prevents instantiation of this class."""
        raise NotImplementedError("Parameters is an abstract class and cannot be instantiated directly.")

    # Additional methods and class variables should be converted as well.

    def build_robots(self, simulation_yaml: str) -> List[Robot[SE3Pose]]:
        robots = []

        with open(simulation_yaml, "r") as f:
            simulation_data = yaml.safe_load(f)

        for robot_entry in simulation_data["Robots"]:
            # Load environment
            robotConfig = robot_entry["robotConfig"]
            worldConfig = simulation_data["worldConfig"]
            mprimpath = robotConfig["mprimpath"]
            env = self.build_environment(worldConfig, mprimpath)

            # Load robot state
            initial_state = robot_entry["initialState"]
            state = self.build_SE3Pose(initial_state)

            ID = robot_entry["ID"]
            # Load Target Model
            infoModel = self.build_robot_TMM(self.targetConfig, ID)
            # Load the sensors
            sensor = self.build_sensor(robotConfig["sensorConfig"])
            # Finally construct the robots
            robots.append(Robot(state, env, infoModel, sensor))

        return robots


    def build_se3pose(input: np.ndarray) -> SE3Pose:
        position = np.array([input[0], input[1], input[2]])
        orientation = rotz(input[3])
        return SE3Pose(position, orientation)
    
    
    def build_planner(planner_yaml: str):
        with open(planner_yaml, 'r') as f:
            planner_node = yaml.safe_load(f)
    
        type_ = planner_node["plannerType"]
        if type_ == "ARVI":
            return build_planner_arvi(planner_yaml)
        else:
            raise ValueError("Unknown planner type. Try a valid type, e.g. ARVI, MCTS.")
    
    
    def build_planner_arvi(planner_yaml: str):
        with open(planner_yaml, 'r') as f:
            planner_node = yaml.safe_load(f)
    
        T = planner_node["T"]
        eps = planner_node["eps"]
        delta = planner_node["del"]
        alloc_time = planner_node["allocated_time"]
        comm_range = planner_node["comm_range"]
    
        planner = InfoPlanner(T, eps, delta, comm_range, alloc_time)
        return planner
    
    
    def build_robot_tmm(target_yaml: str, ID: int):
        with open(target_yaml, 'r') as f:
            target_node = yaml.safe_load(f)
    
        type_ = target_node["target_model"]
        if type_ == "normal":
            return build_info_tmm(target_yaml)
        else:
            raise ValueError("Unknown target Model type. Try a valid type, e.g. normal, adversarial.")
    
    
    def build_info_tmm(target_yaml: str):
        tmm = build_tmm(target_yaml)
        # TODO: Implement the rest of the function.
    
    def build_tmm(target_yaml: str):
        # TODO: Implement the function.
    
    def build_sensor(sensor_model: str):
        with open(sensor_model, 'r') as f:
            sensor_node = yaml.safe_load(f)
    
        type_ = sensor_node["sensorType"]
        if type_ == "range_bearing":
            return build_sensor_range_bearing(sensor_model)
        else:
            raise ValueError("Unknown Sensor type. Try a valid type, e.g. range_bearing.")
    
    def build_sensor_range_bearing(sensor_model: str):
        with open(sensor_model, 'r') as f:
            sensor_node = yaml.safe_load(f)
    
        r_sense = sensor_node["r_sense"]
        fov = sensor_node["fov"]
        b_sigma = sensor_node["b_sigma"]
        r_sigma = sensor_node["r_sigma"]
    
        sensor = RangeBearingSensor(r_sense, fov, b_sigma, r_sigma)
        return sensor
    
    def build_environment(env_model: str, mprim_yaml: str):
        with open(env_model, 'r') as f:
            yaml_env = yaml.safe_load(f)
    
        # TODO: Implement the rest of the function.