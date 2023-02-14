//
// Created by brent on 8/21/18.
//

#ifndef INFO_GATHERING_TARGETMODEL_H
#define INFO_GATHERING_TARGETMODEL_H

#include <Eigen/Dense>
#include <map>
#include <igl/utils/utils_nx.h>
#include <iostream>
#include <igl/mapping/map_nx.h>

using namespace Eigen;

namespace nx {

/**
 * Virtual class for an abstract Target type. Used to simulate temporal processes in an environment.
 */
class Target {

 public:
  int ID; // Target has an ID
  int y_dim; // State dimension

  /**
   * @brief Initialize a target with the identifier and state dimension given.
   * @param ID The identification number of the target.
   * @param y_dim The state dimension of the target.
   */
  Target(int ID, int y_dim) : ID(ID), y_dim(y_dim) {}

  /**
   * @brief Returns the 3D position of a point target.
   * @return The 3D position.
   */
  virtual Vector3d getPosition() const = 0;

  /**
   * @brief Returns the state of interest about the target.
   * @return The target state being tracked.
   */
  virtual VectorXd getState() const = 0;

  /**
   * @brief Returns the Jacobian of the system dynamics about the current target state.
   * @return The Jacobian matrix.
   */
  virtual MatrixXd getJacobian() const = 0;

  /**
   * @brief Returns the Noise matrix at the current target state.
   * @return The noise matrix.
   */
  virtual MatrixXd getNoise() const = 0;

  /**
   * @brief Simulates T forward steps of the target dynamics.
   * @param T the number of timesteps to simulate.
   * @param map Used for boundary checking.
   * @param cmap Used for collision checking.
   */
  virtual void forwardSimulate(int T, const nx::map_nd &map, const std::vector<char> &cmap) = 0;

}; // End Target Class

/**
 * Generic Target Model. This model is capable of simulating a ground truth target, as well as providing Jacobians
 * that the robot may use.
 */
class TargetModel {

 public:
  int target_dim{0}; // Overall dimension of the system.
  std::map<int, std::shared_ptr<Target>> targets; // Dictionary of target types, internally stored as shared pointers.
  std::shared_ptr<nx::map_nd> map; // Map of the environment for boundary checking.
  const std::vector<char> cmap;

  /**
   * Constructs an empty Target Model with a map.
   * @param map The map parameters of the environment.
   */
  TargetModel(const nx::map_nd &map, const std::vector<char> &cmap) :
      target_dim(0),
      map(std::make_shared<nx::map_nd>(map)),
      cmap((cmap)) {};

  /**
   * Adds a target to the Target Model. Returns false if a target with the same ID has been already added.
   * @param ID The ID of the target to add.
   * @param target The Target to be added.
   * @return The result of the AddTarget operation.
   */
  bool addTarget(int ID, std::shared_ptr<Target> target) {
    return AddSharedTarget(ID, target);
  }

//  /**
//   * Updates the state of the target with specified ID. Does nothing if the target does not exist.
//   * @param ID The ID of the target to be updated.
//   * @param state The new state to be updated.
//   */
//  void updateTarget(int ID, const VectorXd &state) {
//    targets.at(ID)->state = state;
//  }

  /**
   * Removes a target with the ID specified.
   * @param ID The ID to be removed from the target set.
   */
  void removeTarget(int ID) {
    std::shared_ptr<Target> target = targets[ID];
    target_dim -= target->y_dim;
    targets.erase(ID);
  }

  /**
   * Returns the joint target state as a Vector.
   * @return The target state to be returned.
   */
  VectorXd getTargetState() const {
    VectorXd result = VectorXd::Zero(target_dim);
    // Loop over all targets to construct the joint state as a Vector.
    int index = 0;
    for (const auto &pair: targets) {
      std::shared_ptr<Target> target = pair.second;
      result.segment(index, target->y_dim) = target->getState();
      index += target->y_dim;
    }
    // Return the result.
    return result;
  }

  /**
   * Returns the joint system matrix A of the target model.
   * @return The system matrix.
   */
  MatrixXd getSystemMatrix() const {
    MatrixXd result = MatrixXd::Zero(target_dim, target_dim);
    // Loop over all targets to construct the joint system matrix.
    int index = 0;
    for (const auto &pair: targets) {
      std::shared_ptr<Target> target = pair.second;
      result.block(index, index, target->y_dim, target->y_dim) = target->getJacobian();
      index += target->y_dim;
    }
    // Return the result.
    return result;
  }

  /**
   * Returns the joint noise matrix W of the target model.
   * @return The noise matrix.
   */
  MatrixXd getNoiseMatrix() const {
    MatrixXd result = MatrixXd::Zero(target_dim, target_dim);
    // Loop over all targets to construct the joint system matrix.
    int index = 0;
    for (const auto &pair: targets) {
      std::shared_ptr<Target> target = pair.second;
      result.block(index, index, target->y_dim, target->y_dim) = target->getNoise();
      index += target->y_dim;
    }
    // Return the result.
    return result;
  }

  /**
   * Updates the state of all targets by simulating the gaussian system model.
   * @param T The number of timesteps to evolve the environment by.
   */
  void forwardSimulate(int T) {
    // Update each target's state.
    for (auto &pair : targets)
      pair.second->forwardSimulate(T, *map, cmap);
  }

  /**
   * Returns the number of targets being modeled.
   * @return Number of targets.
   */
  unsigned long num_targets() const {
    return targets.size();
  }

  /**
  * Returns the Target by ID
  * @param ID The ID of the target to return.
  * @return The Target queried for.
  */
  std::shared_ptr<Target> getTargetByID(int ID) const {
    return targets.at(ID);
  }

 protected:

  /**
   * Underlying shared_ptr implementation of the AddTarget function.
   * @param ID The ID to be added.
   * @param target_ptr The shared_ptr to the Target object.
   * @return The result of the Add operation.
   */
  bool AddSharedTarget(int ID, std::shared_ptr<Target> target_ptr) {
    bool result = false;
    if (!targets.count(ID)) // Target already in list
    {
      targets.emplace(ID, target_ptr);
      target_dim += target_ptr->y_dim; // Increment the target dimension.
      result = true;
    }
    return result;
  }

}; // End Class

} // End Namespace

#endif //INFO_GATHERING_TARGETMODEL_H
