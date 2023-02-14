//
// Created by brent on 2/19/18.
//

#ifndef INFO_GATHERING_PARAMS_H_H
#define INFO_GATHERING_PARAMS_H_H

#include <vector>
#include <string>
//#include <igl/sensing/observation_models/range_sensor.h>
//#include <igl/sensing/observation_models/camera_sensor.h>
#include <igl/sensing/observation_models/range_bearing_sensor.h>
#include <igl/target_models/target_model.h>
#include <igl/target_models/info_target_model.h>
#include <igl/target_models/dynamics/static_2d.h>
#include <igl/target_models/dynamics/double_integrator_2d.h>
#include <igl/target_models/belief/static_2d_belief.h>
#include <igl/target_models/belief/double_integrator_2d_belief.h>
#include <igl/env/env_int.h>
#include <igl/env/env_se2.h>
#include <igl/robot.h>
#include <igl/planning/infoplanner.h>
#include <yaml-cpp/yaml.h>
#include <igl/utils/utils_nx.h>

namespace nx {

/**
 * The Parameters class is capable of reading in high level descriptions of a multi-robot Information Gathering
 * simulation, and configuring the team of robots, target environment, and a planner for solving information gathering
 * tasks. This class is the preferred way to generate simulations in this library.
 */
class Parameters {

 public:

  // Global Simulation Properties
  std::string plannerConfig{""};
  std::string targetConfig{""};
  double samp{0.5};
  int n_controls{1};
  double debug{0};
  int log{0};
  int Tmax{10};
  std::string outputPath{""};

  /**
   * Loads all simulation parameters from a top level YAML file.
   * @param yaml_file
   */
  Parameters(std::string yaml_file) {

    YAML::Node node = YAML::LoadFile(yaml_file); // Root node
    plannerConfig = node["plannerConfig"].as<std::string>();
    targetConfig = node["targetConfig"].as<std::string>();
    samp = node["samp"].as<double>();
    n_controls = node["n_controls"].as<int>();
    Tmax = node["Tmax"].as<int>();
    log = node["log"].as<int>();
    debug = node["debug"].as<int>();
    outputPath = node["outputpath"].as<std::string>();

    // Build Simulation Actors
    robots = BuildRobots(yaml_file); // Robots load file from
    p = BuildPlanner(plannerConfig);
    world = BuildTMM(targetConfig);
  }

  /**
   * Returns the vector of simulation robots.
   * @return The vector of robots.
   */
  std::vector<Robot<SE3Pose>> GetRobots() const {
    return robots;
  }
  /**
   * Returns the construted planner for the simulation.
   * @return The constructed planner.
   */
  std::shared_ptr<InfoPlanner<SE3Pose>> GetPlanner() const {
    return p;
  }

  /**
   * Returns the target motion model.
   * @return The target motion model.
   */
  std::shared_ptr<TargetModel> GetTMM() const {
    return world;
  }

 protected:

  /**
   * Default Constructor
   */
  Parameters() {}

  std::vector<Robot<SE3Pose>> robots;
  std::shared_ptr<InfoPlanner<SE3Pose>> p;
  std::shared_ptr<TargetModel> world;
  std::shared_ptr<nx::map_nd> map;
  std::shared_ptr<std::vector<char>> cmap;

  /**
   * Builds a set of robots configured for the desired simulation.
   * @param simulation_yaml The high level scenario file describing the simulation.
   * @return A vector of constructed robots.
   */
  std::vector<Robot<SE3Pose>> BuildRobots(std::string simulation_yaml) {
    std::vector<Robot<SE3Pose>> robots;

    for (const auto &robot_entry : YAML::LoadFile(simulation_yaml)["Robots"]) {

      // Load environment
      YAML::Node robotConfig = YAML::LoadFile(robot_entry["robotConfig"].as<std::string>());
      std::shared_ptr<Environment<SE3Pose>> env =
          BuildEnvironment(
              YAML::LoadFile(simulation_yaml)["worldConfig"].as<std::string>(),
              robotConfig["mprimpath"].as<std::string>());

      // Load robot state

      std::vector<double> initialState = robot_entry["initialState"].as<std::vector<double>>();
      if (debug)
        std::cout << "loading robot params initial state: " << initialState[0] << initialState[1] << initialState[2]
                  << std::endl;
      SE3Pose state = BuildSE3Pose(initialState);
      if (debug)
        std::cout << "verify initial state: " << state.position[0] << state.position[1] << state.position[2]
                  << std::endl;

      int ID = robot_entry["ID"].as<int>();
      // Load Target Model
      std::shared_ptr<InfoTargetModel> infoModel = BuildRobotTMM(targetConfig, ID);
      std::cout <<"Loaded Target Model\n";
      // Load the sensors
      std::shared_ptr<Sensor<SE3Pose>> sensor = BuildSensor(robotConfig["sensorConfig"].as<std::string>());
      std::cout <<"Loaded Sensor Model\n";
      // Finally construct the robots
      robots.push_back(Robot<SE3Pose>(state, env, infoModel, sensor));
      std::cout <<"Loaded Robots\n";
    }

    return robots;
  }

  /**
   * Builds an SE3Pose object from an (X,Y, Z, Yaw) pair.
   * @return The desired SE3Pose object.
   */
  SE3Pose BuildSE3Pose(const std::vector<double> &input) {
    Vector3d position;
    position << input[0], input[1], input[2];
    Matrix3d orientation = rotz(input[3]);
    return SE3Pose(position, orientation);
  }

  /**
   * Builds a generic Planner type from a YAML configuration file.
   * @param planner_yaml String path to the planner configuration file.
   * @return A pointer to the constructed Planner.
   */
  std::shared_ptr<InfoPlanner<SE3Pose>> BuildPlanner(std::string planner_yaml) {

    YAML::Node planner_node = YAML::LoadFile(planner_yaml);
    std::string type = planner_node["plannerType"].as<std::string>();
    if (type == "ARVI") {
      return BuildPlannerARVI(planner_yaml);
    } else {
      std::cerr << "Unknown planner type. Try a valid type, e.g. ARVI, MCTS.";
    }
  }

  /**
   * Builds an ARVI Planner from a YAML configuration file.
   * @param planner_yaml String path to the YAML configuration file.
   * @return A pointer to the constructed Planner.
   */
  std::shared_ptr<InfoPlanner<SE3Pose>> BuildPlannerARVI(std::string planner_yaml) {
    YAML::Node planner_node = YAML::LoadFile(planner_yaml);
    int T = planner_node["T"].as<int>();
    double eps = planner_node["eps"].as<double>();
    double del = planner_node["del"].as<double>();
    double alloc_time = planner_node["allocated_time"].as<double>();
    double comm_range = planner_node["comm_range"].as<double>();
    std::shared_ptr<InfoPlanner<SE3Pose>>
        p(new InfoPlanner<SE3Pose>());// del, eps, comm_range, alloc_time, debug));
    return p;
  }

  /**
   * Builds a Robots Target Motion Model.
   * @param target_yaml The string of the Target Model to be configured.
   * @param ID ID number of the Robot being built for.
   * @return A pointer to the constructed Target Motion Model.
   */
  std::shared_ptr<InfoTargetModel> BuildRobotTMM(std::string target_yaml, int ID) {

    YAML::Node target_node = YAML::LoadFile(target_yaml);
    std::string type = target_node["target_model"].as<std::string>();
    if (type == "normal") {
      return BuildInfoTMM(target_yaml);
    } else {
      std::cerr << "Unknown target Model type. Try a valid type, e.g. normal, adversarial.";
    }
  }

  /**
   * Builds an InfoTargetModel, which is an extension of the Target Motion Model containing a belief state.
   * @param target_yaml The YAML file of the InfoTMM to be built.
   * @return A pointer to the Info Target Model.
   */
  std::shared_ptr<InfoTargetModel> BuildInfoTMM(std::string target_yaml) {

    std::shared_ptr<TargetModel> tmm = BuildTMM(target_yaml); // Parse the target model information.
    YAML::Node tmm_node = YAML::LoadFile(target_yaml);
    // Read in YAML Parameters
    int use_prior = tmm_node["use_prior"].as<int>();
    int velocity = tmm_node["velocity"].as<int>();
    int num_targets = tmm->num_targets();
    int dim = tmm->target_dim / num_targets;
    double q = tmm_node["q"].as<double>();
    double cov_pos = tmm_node["cov_pos"].as<double>();
    double cov_vel = tmm_node["cov_vel"].as<double>();
    SystemModel m = systemMatrix(q, dim, cov_pos, velocity, cov_vel); // System Dynamics

    // Build InfoTargetModel
    map_nd map_ = *map;
    std::shared_ptr<InfoTargetModel> model = std::make_shared<InfoTargetModel>(InfoTargetModel(map_, *cmap));
    if (use_prior) { // Add target priors!
      for (const auto &pair : tmm->targets) {
        int ID = pair.first;
        auto target = pair.second;
        if (!velocity)
          model->addTarget(ID, std::make_shared<Static2DBelief>(Static2D(ID, target->getPosition().head(2), q), m.Sigma));
        else
          model->addTarget(ID, std::make_shared<DoubleIntegrator2DBelief>(
              DoubleIntegrator2D(ID, target->getState().head(2), target->getState().tail(2), samp, 1.0, q), m.Sigma));

      }
    }
    std::cout <<"Initial Target Covariance: " << model->getCovarianceMatrix() << std::endl;

    return model;
  }

  /**
   * Builds the Target Motion Model from a YAML  configuration file.
   * @param target_yaml The path to the YAML configuration file.
   * @return A pointer to the constructed Target Motion Model.
   */
  std::shared_ptr<TargetModel> BuildTMM(std::string target_yaml) {
    // TODO Add checking for correct DA / Target size
    // Load YAML Parameters
    YAML::Node tmm_node = YAML::LoadFile(target_yaml);
    int velocity = tmm_node["velocity"].as<int>();
    double cov_pos = tmm_node["cov_pos"].as<double>();
    double cov_vel = tmm_node["cov_vel"].as<double>();
    int dim = tmm_node["target_dim"].as<int>(); // Dimension per target.
    double q = tmm_node["q"].as<double>();
    const std::vector<int> &env_da = tmm_node["env_da"].as<std::vector<int>>();
    MatrixXd y_mat = nx::yamlInitMatrix(tmm_node["y0"].as<std::vector<std::vector<double>>>());
    VectorXd y = stateMatToVector(y_mat, velocity, dim);

    // Construct the Target Model
    map_nd map_ = *map;
    std::shared_ptr<TargetModel> tmm = std::make_shared<TargetModel>(TargetModel(map_, *cmap));
    std::vector<int> da = env_da;
    for (int i = 0; i < da.size(); i++) {
      int ID = da[i];
      VectorXd state = y.segment(dim * i, dim);
      if (!velocity)
        tmm->addTarget(ID, std::make_shared<Static2D>(ID, state, q));
      else
        tmm->addTarget(ID, std::make_shared<DoubleIntegrator2D>(ID, state.head(2), state.tail(2), samp, 1.0, q));
    }
    return tmm;
  }

  /**
   * Builds a generic Sensor type from a YAML configuration file.
   * @param sensor_model The YAML containing the specification of the desired sensor.
   * @return The sensor constructed from the file.
   */
  std::shared_ptr<Sensor<SE3Pose>> BuildSensor(std::string sensor_model) {

    YAML::Node sensor_node = YAML::LoadFile(sensor_model);
    std::string type = sensor_node["sensorType"].as<std::string>();
    if (type == "range_bearing") {
      return BuildSensorRangeBearing(sensor_model);
// TODO Add back the range only and camera sensors.
      //    } else if (type == "range") {
//      return BuildSensorRangeOnly(sensor_model);
//    } else if (type == "camera") {
//      return BuildCamera(sensor_model);
    } else {
      std::cerr << "Unknown Sensor type. Try a valid type, e.g. range_bearing.";
    }
  }

  /**
   * Constructs a Range-Bearing sensor model from a YAML configuration file.
   * @param sensor_model The path to the filename of the sensor configuration.
   * @return A pointer to the constructed Range-Bearing sensor.
   */
  std::shared_ptr<Sensor<SE3Pose>> BuildSensorRangeBearing(std::string sensor_model) {

    YAML::Node sensor_node = YAML::LoadFile(sensor_model);
    std::shared_ptr<Sensor<SE3Pose>> sensor;
    // Initialize Sensor Observation Model
    double r_sense = sensor_node["r_sense"].as<double>();
    double fov = sensor_node["fov"].as<double>();
    double b_sigma = sensor_node["b_sigma"].as<double>();
    double r_sigma = sensor_node["r_sigma"].as<double>();

    sensor = std::make_shared<nx::RangeBearingSensor>(nx::RangeBearingSensor(r_sense, fov, b_sigma, r_sigma, map, *cmap));
    return sensor;
  }

  /**
   * Constructs a Range-Only sensor model from a YAML configuration file
   * @param sensor_model The path to the filename of the sensor configuration.
   * @return A pointer to the constructed Range-Only sensor.
   */
//  std::shared_ptr<Sensor<SE3Pose>> BuildSensorRangeOnly(std::string sensor_model) {
//
//    YAML::Node sensor_node = YAML::LoadFile(sensor_model);
//    std::shared_ptr<Sensor<SE3Pose>> sensor;
//    // Initialize Sensor Observation Model
//    double min_range = sensor_node["min_range"].as<double>();
//    double max_range = sensor_node["max_range"].as<double>();
//    double min_hang = sensor_node["min_hang"].as<double>();
//    double max_hang = sensor_node["max_hang"].as<double>();
//    double min_vang = sensor_node["min_vang"].as<double>();
//    double max_vang = sensor_node["max_vang"].as<double>();
//    double noise_stdev = sensor_node["noise_stdev"].as<double>();
//
//    std::string off_file; // Check for Off-file
//    try {
//      off_file = sensor_node["off_file"].as<std::string>();
//    }
//    catch (std::exception e) {}
//    sensor.reset(new nx::RangeSensor(min_range, max_range,
//                                     min_hang, max_hang,
//                                     min_vang, max_vang,
//                                     noise_stdev, off_file));
//    return sensor;
//  }

  /**
   * Constructs a Camera sensor from a given YAML configuration file.
   * @param sensor_model The path to the filename of the sensor configuration.
   * @return A pointer to the constructed Camera Sensor.
   */
//  std::shared_ptr<Sensor<SE3Pose>> BuildCamera(std::string sensor_model) {
//
//    YAML::Node sensor_node = YAML::LoadFile(sensor_model);
//    std::shared_ptr<Sensor<SE3Pose>> sensor;
//    // Initialize Sensor Observation Model
//    double height = sensor_node["height"].as<double>();
//    double width = sensor_node["width"].as<double>();
//    MatrixXd K = nx::yamlInitMatrix(sensor_node["K"].as<std::vector<std::vector<double>>>());
//    std::string off_file = sensor_node["off_file"].as<std::string>();
//
//    double noise_stdev = sensor_node["noise_stdev"].as<double>();
//
//    sensor.reset(new nx::CameraSensor(height, width, K, noise_stdev, off_file));
//    return sensor;
//  }

  /**
   * Builds the environment for individual robots, based on a global environment configuration and the
   * robot's unique motion primitives.
   * @param envModel The global environment configuration.
   * @param mprim_yaml The robots unique motion primitives.
   * @return A pointer to the environment data structure.
   */
  std::shared_ptr<Environment<SE3Pose>> BuildEnvironment(std::string env_model, std::string mprim_yaml) {

    YAML::Node yaml_env = YAML::LoadFile(env_model);

    // Build MAP_ptr
    YAML::Node envConfig = YAML::LoadFile(env_model);
    map = std::make_shared<nx::map_nd>(nx::map_nd());
    cmap = std::make_shared<std::vector<char>>(std::vector<char>());
    map->initFromYaml(*cmap, env_model);
    return std::shared_ptr<Environment<SE3Pose>>(new SE2Environment(*map, *cmap, mprim_yaml));
  }

  struct SystemModel {
    MatrixXd A;
    MatrixXd W;
    MatrixXd Sigma;
  };

  SystemModel systemMatrix(double q, int dim, double cov_pos, int velocity = 0, double cov_vel = -1) {
    SystemModel m;
    if (velocity) {
      m.A = MatrixXd::Identity(4, 4);
      m.W = MatrixXd::Identity(4, 4);
      m.Sigma = MatrixXd::Zero(4, 4);
      // Set Dynamics matrix
      m.A.block(0, 2, 2, 2) = samp * MatrixXd::Identity(2,2);
      // Set Noise matrix
      m.W.block(0, 0, 2, 2) = samp * samp * samp / 3 * MatrixXd::Identity(2, 2);
      m.W.block(0, 2, 2, 2) = samp * samp / 2 * MatrixXd::Identity(2, 2);
      m.W.block(2, 0, 2, 2) = samp * samp / 2 * MatrixXd::Identity(2, 2);
      m.W.block(2, 2, 2, 2) = samp * MatrixXd::Identity(2, 2);
      m.W = m.W * q; // Apply diffusion strength
      // Set Covariance Matrix
      m.Sigma.block(0, 0, 2, 2) = cov_pos * MatrixXd::Identity(2, 2);
      m.Sigma.block(2, 2, 2, 2) = cov_vel * MatrixXd::Identity(2, 2);
    } else { // Position only, e.g. Random Walk
      m.A = MatrixXd::Identity(2, 2);
      m.W = q * MatrixXd::Identity(2, 2);
      m.Sigma = cov_pos * MatrixXd::Identity(2, 2);
    }
    return m;
  }

  VectorXd stateMatToVector(MatrixXd y, int velocity, int target_dim) {
    if (velocity) {
      // First check if we need to append velocities
      // If needed, add 0 initial velocity to the target state
      if (y.cols() == 2) {
        MatrixXd y_temp(y.rows(), 4);
        y_temp.setZero();
        y_temp.block(0, 0, y.rows(), 2) = y;
        y.resize(y_temp.rows(), y_temp.cols());
        y = y_temp;
      }
    }
    int n_targets = y.rows();
    // Now construct the vector.
    VectorXd output(target_dim * n_targets);
    output.fill(0.0);
    // Append all targets
    for (int i = 0; i < n_targets; i++) {
      output.segment(i * target_dim, target_dim) = y.row(i);
    }
    return output;
  }

};
} // end namespace

#endif //INFO_GATHERING_PARAMS_H_H
