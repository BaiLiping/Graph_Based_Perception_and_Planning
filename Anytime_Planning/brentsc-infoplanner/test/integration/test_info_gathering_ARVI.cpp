// Author: Brent Schlotfeldt
// Usage:
//    ../bin/test_info_planner_SE2 ../data/test_one_robot_ARVI.yaml
//
// Profile:
//  Compile with Debug info
//  valgrind --tool=callgrind ../bin/test_info_planner_SE2 ../data/test_one_robot_ARVI.yaml
//  kcachegrind callgrind.out.6000
//
//

#include <igl/env/env_se2.h>
#include <igl/planning/infoplanner.h>
#include <igl/mapping/map_nx.h>
#include <igl/utils/utils_nx.h>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <igl/robot.h>
#include <igl/params.h>
#include <igl/estimation/kalman_filter.h>
#include <igl/se3_pose.h>
#include <igl/estimation/multi_target_filter.h>

using namespace Eigen;

int main(int argc, char **argv) {

  nx::Parameters p(argv[1]); // Load parameters from the file given.

  auto robots = p.GetRobots();
  auto planner = p.GetPlanner();
  auto world = p.GetTMM();

  std::vector<std::vector<VectorXd>> fixed_traj(robots.size());
  std::vector<nx::PlannerOutput<nx::SE3Pose>> plannerOutputs(robots.size());
  // Main Loop

  auto t1 = nx::tic();

  for (int t = 0; t < p.Tmax; t += 1) {
    std::cout <<"Timestep: " << t << std::endl;

    // Sense, and Filter.
    for (int i = 0; i < robots.size(); i++) {
      std::vector<nx::Measurement> m = robots[i].sensor->senseMultiple(robots[i].getState(), world);
      nx::GaussianBelief output = nx::MultiTargetFilter<nx::SE3Pose>::MultiTargetKF(m, robots[i]);
      robots[i].tmm->updateBelief(output.mean, output.cov);
    }

    // PlanARVI (Every n_controls steps)
    if (t % p.n_controls == 0) {
      std::cout << "Starting InfoPlanner..." << std::endl;
      // Reset planning variables
      for (int i = 0; i < robots.size(); i++) {
        fixed_traj[i].clear();
      }
      for (int i = 0; i < robots.size(); i++) {
        plannerOutputs[i] = planner->PlanARVI(robots[i], 12);
        // Record fixed trajectories for Coordinate Descent
        int t_ = 0;
        for (auto it = plannerOutputs[i].path.begin();
             it != plannerOutputs[i].path.end(); ++t_, ++it) {
          VectorXd point = robots[i].env->state_to_SE2(*it);
          //std::cout << "robot i: " << point << std::endl;
          fixed_traj[i].push_back(point);
        }
      }
      std::cout << "Computation done in " << nx::toc(t1) << " sec!" << std::endl;
      // TODO Here we should update the collision map, to avoid robots crossing trajectories.
    }

    // Actuate
    for (int i = 0; i < robots.size(); i++) {
      // Apply controls
      robots[i].applyControl(plannerOutputs[i].action_idx, 1);
    }
    // Environment Changes
    world->forwardSimulate(1);
  }

  std::cout << "Final ground truth is: " << world->getTargetState() << std::endl;

  for (int i = 0; i < robots.size(); i++) {
    std::cout << "Final Estimate_" << i << " is: " << robots[i].tmm->getTargetState() << std::endl
              << "Final cov is: "
              << robots[i].tmm->getCovarianceMatrix() << std::endl;
    std::cout << "Final pose_" << i << " is: " << robots[i].getState().position << std::endl;
  }
  return 0;
}

