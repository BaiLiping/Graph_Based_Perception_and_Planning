#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <igl/params.h>
#include <igl/sensing/observation_models/range_sensor.h>
#include <igl/sensing/observation_models/bearing_sensor.h>
#include <igl/sensing/observation_models/position_sensor.h>
#include <igl/robot.h>
#include <igl/estimation/kalman_filter.h>
#include <igl/planning/infoplanner.h>
#include <Eigen/Dense>
#include <vector>
#include <igl/estimation/multi_target_filter.h>

#include <igl/planning/cost_functions/trace_cost.h>
#include <igl/planning/cost_functions/determinant_cost.h>

#include <igl/target_models/dynamics/static_2d.h>
#include <igl/target_models/dynamics/se2_target.h>
#include <igl/target_models/belief/static_2d_belief.h>

namespace py = pybind11;
using namespace nx;
using namespace Eigen;

/**
 * Python Bindings for Infomation Gathering Library.
 */
PYBIND11_MODULE(pyInfoGathering, m) {

/***************************************************************************************************
                        Mapping and Robot Models
***************************************************************************************************/
  // Map
  py::class_<map_nd, std::shared_ptr<map_nd>>(m, "map_nd")
      .def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>())
      .def("min", &map_nd::min)
      .def("max", &map_nd::max)
      .def("res", &map_nd::res)
      .def("size", &map_nd::size);

  // SE3Pose Binding
  py::class_<SE3Pose>(m, "SE3Pose")
      .def(py::init<const Vector3d &, const Vector4d &>())
      .def(py::init<const Vector3d &, const Matrix3d &>())
      .def("getYaw", &SE3Pose::getYaw)
      .def("getPitch", &SE3Pose::getPitch)
      .def("getRoll", &SE3Pose::getRoll)
      .def("getSE2", &SE3Pose::getSE2)
      .def_readonly("position", &SE3Pose::position)
      .def_readonly("orientation", &SE3Pose::orientation);

  // Environment
  py::class_<Environment<SE3Pose>, std::shared_ptr<Environment<SE3Pose>>>(m, "Environment")
      .def_readonly("map", &Environment<SE3Pose>::map)
      .def("cmap", &Environment<SE3Pose>::GetCostMap);


  // SE2 Environment
  py::class_<SE2Environment, std::shared_ptr<SE2Environment>>(m, "SE2Environment")
      .def(py::init<const nx::map_nd &, const std::vector<char> &, const std::string &>())
      .def_readonly("map", &Environment<SE3Pose>::map)
      .def("cmap", &Environment<SE3Pose>::GetCostMap);

/***************************************************************************************************
                        Target Models
***************************************************************************************************/

  py::class_<Target, std::shared_ptr<Target>>(m, "Target")
      .def("getPosition", &Target::getPosition)
      .def_readonly("ID", &Target::ID);

  // Target Model
  py::class_<TargetModel, std::shared_ptr<TargetModel>>(m, "target_model")
      .def(py::init<const nx::map_nd &, const std::vector<char> &>())
      .def("addTarget", &TargetModel::addTarget)
      .def("getTargetState", &TargetModel::getTargetState)
      .def("getTargetByID", &TargetModel::getTargetByID)
      .def("forwardSimulate", &TargetModel::forwardSimulate)
      .def_readonly("targets", &TargetModel::targets);

  // InfoTarget
  py::class_<InfoTarget, Target, std::shared_ptr<InfoTarget>>(m, "InfoTarget")
      .def("getCovariance", &InfoTarget::getCovariance)
      .def_readonly("ID", &InfoTarget::ID);

  // Info Target Model
  py::class_<InfoTargetModel, TargetModel, std::shared_ptr<InfoTargetModel>>(m, "info_target_model")
      .def(py::init<const nx::map_nd &, const std::vector<char> &>())
      .def("addTarget", &InfoTargetModel::addTarget)
      .def("updateBelief", &InfoTargetModel::updateBelief)
      .def("getCovarianceMatrix", &InfoTargetModel::getCovarianceMatrix)
      .def("getTargetByID", &InfoTargetModel::getTargetByID)
      .def_readonly("targets", &InfoTargetModel::targets);

  // Static 2D
  py::class_<Static2D, Target, std::shared_ptr<Static2D>>(m, "Static2D")
      .def(py::init<int, const Vector2d &, double>());
  py::class_<Static2DBelief, Static2D, InfoTarget, std::shared_ptr<Static2DBelief>>(m, "Static2DBelief")
      .def(py::init<const Static2D &, const MatrixXd &>());
  // Double Integrator
  py::class_<DoubleIntegrator2D, Target, std::shared_ptr<DoubleIntegrator2D>>(m, "DoubleInt2D")
      .def(py::init<int, const Vector2d &, const Vector2d &, double, double, double>());
  py::class_<DoubleIntegrator2DBelief, DoubleIntegrator2D, InfoTarget,
             std::shared_ptr<DoubleIntegrator2DBelief>>(m, "DoubleInt2DBelief")
      .def(py::init<const DoubleIntegrator2D &, const MatrixXd &>());

  // SE2 Target
  py::class_<SE2Target, Target, std::shared_ptr<SE2Target>>(m, "SE2Target")
      .def(py::init<int, const Vector3d &, std::shared_ptr<ControlPolicy<3, 2>>, double, double >(),
          py::arg("ID"), py::arg("se2_pose"), py::arg("policy"), py::arg("tau"), py::arg("q"));

  // Control Policy
  py::class_<ControlPolicy<3, 2>, std::shared_ptr<ControlPolicy<3, 2>>>(m, "SE2Policy")
      .def(py::init<std::function<VectorXd(const VectorXd &)>>())
      .def("computeControl", &ControlPolicy<3, 2>::computeControl);


/***************************************************************************************************
                        Observation Models
***************************************************************************************************/

  // Sensing and Estimation
  py::class_<Sensor<SE3Pose>, std::shared_ptr<Sensor<SE3Pose>>>(m, "Sensor")
      .def("senseMultiple", &Sensor<SE3Pose>::senseMultiple)
      .def("sense", &Sensor<SE3Pose>::sense);

  // RangeBearing Sensor
  py::class_<RangeBearingSensor, Sensor<SE3Pose>, std::shared_ptr<RangeBearingSensor>>(m, "RangeBearingSensor")
      .def(py::init<double, double, double, double, std::shared_ptr<nx::map_nd>, const std::vector<char> &>());
  // Range-Only Sensor
  py::class_<RangeSensor, Sensor<SE3Pose>, std::shared_ptr<RangeSensor>>(m, "RangeSensor")
      .def(py::init<double, double, double, double, double, double, double,
                    std::shared_ptr<nx::map_nd>, const std::vector<char> &>());
  // Bearing-Only Sensor
  py::class_<BearingSensor, Sensor<SE3Pose>, std::shared_ptr<BearingSensor>>(m, "BearingSensor")
      .def(py::init<double, double, double, double, double, std::shared_ptr<nx::map_nd>, const std::vector<char> &>());

  // Position Sensor
  py::class_<PositionSensor, Sensor<SE3Pose>, std::shared_ptr<PositionSensor>>(m, "PositionSensor")
      .def(py::init<double, double, double, double, double, double, double,
                    std::shared_ptr<nx::map_nd>, const std::vector<char> &>());
/***************************************************************************************************
                        State Estimation
***************************************************************************************************/
  // Measurement
  py::class_<Measurement>(m, "Measurement")
      .def_readonly("z", &Measurement::z)
      .def_readonly("ID", &Measurement::ID)
      .def_readonly("validity", &Measurement::valid)
      .def_readonly("z_dim", &Measurement::z_dim);

//  // Kalman Filter
//  m.def("KalmanFilter", (KFOutput(*)(Measurement,
//  const Robot<SE3Pose> &, bool)) &KalmanFilter::KFCovariance, "Kalman Filter",
//      py::arg("m"), py::arg("robot"), py::arg("debug") = false);

  // Multi Target Filter
  m.def("MultiTargetFilter",
        (GaussianBelief (*)(std::vector<Measurement>,
                            const Robot<SE3Pose> &,
                            bool)) &MultiTargetFilter<SE3Pose>::MultiTargetKF,
        "Multiple Target Kalman Filter",
        py::arg("m"),
        py::arg("robot"),
        py::arg("debug") = false);

  // Filter Output
  py::class_<GaussianBelief>(m, "GaussianBelief")
      .def_readonly("mean", &GaussianBelief::mean)
      .def_readonly("cov", &GaussianBelief::cov);


/***************************************************************************************************
                        Robot and Planner
***************************************************************************************************/
  // Robot Class Binding
  py::class_<Robot<SE3Pose>>(m, "Robot")
      .def(py::init<SE3Pose, std::shared_ptr<SE2Environment>, std::shared_ptr<InfoTargetModel>,
                    std::shared_ptr<Sensor<SE3Pose>>>())
      .def("getState", &Robot<SE3Pose>::getState)
      .def("applyControl", &Robot<SE3Pose>::applyControl)
      .def_readonly("env", &Robot<SE3Pose>::env)
      .def_readonly("tmm", &Robot<SE3Pose>::tmm)
      .def_readonly("sensor", &Robot<SE3Pose>::sensor);

  // Planner
  py::class_<InfoPlanner<SE3Pose>, std::shared_ptr<InfoPlanner<SE3Pose>>>(m, "InfoPlanner")
      .def(py::init<>())
      .def(py::init<std::shared_ptr<CostFunction>>())
      .def("planAstar", &InfoPlanner<SE3Pose>::PlanAstar)
      .def("computeHeuristic", &InfoPlanner<SE3Pose>::computeHeuristic)
      .def("planFVI", &InfoPlanner<SE3Pose>::PlanFVI)
      .def("planRVI", &InfoPlanner<SE3Pose>::PlanRVI)
      .def("planARVI", &InfoPlanner<SE3Pose>::PlanARVI);

  // PlannerOutput
  py::class_<PlannerOutput<SE3Pose>>(m, "PlannerOutput")
      .def_readonly("path", &PlannerOutput<SE3Pose>::path)
      .def_readonly("cost_per_node", &PlannerOutput<SE3Pose>::cost_per_node)
      .def_readonly("observation_points", &PlannerOutput<SE3Pose>::observation_points)
      .def_readonly("target_path", &PlannerOutput<SE3Pose>::target_path)
      .def_readonly("all_paths", &PlannerOutput<SE3Pose>::all_paths)
      .def_readonly("opened_list", &PlannerOutput<SE3Pose>::opened_list)
      .def_readonly("closed_list", &PlannerOutput<SE3Pose>::closed_list)
      .def_readwrite("action_idx", &PlannerOutput<SE3Pose>::action_idx);

  // Cost Functions
  py::class_<CostFunction, std::shared_ptr<CostFunction>>(m, "CostFunction");

  py::class_<TraceCost, CostFunction, std::shared_ptr<TraceCost>>(m, "TraceCost")
      .def(py::init<>());
  py::class_<DeterminantCost, CostFunction, std::shared_ptr<DeterminantCost>>(m, "DeterminantCost")
      .def(py::init<>());


/***************************************************************************************************
                        Parameters Object
***************************************************************************************************/
  // Parameters Binding
  py::class_<Parameters>(m, "Parameters")
      .def(py::init<const std::string &>())
      .def("GetRobots", &Parameters::GetRobots)
      .def("GetPlanner", &Parameters::GetPlanner)
      .def("GetTMM", &Parameters::GetTMM)
      .def_readonly("samp", &Parameters::samp)
      .def_readonly("n_controls", &Parameters::n_controls)
      .def_readonly("Tmax", &Parameters::Tmax);

}