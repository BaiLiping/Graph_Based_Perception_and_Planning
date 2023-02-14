//
// Created by brentschlotfeldt on 9/19/18.
//

#ifndef ENVIRONMENT_SE3STATE_H
#define ENVIRONMENT_SE3STATE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <exception>
#include <igl/utils/utils_nx.h>

using namespace Eigen;

namespace nx {

/**
 * SE3State stores position and orientation of a rigid body with respect to a world frame.
 */
struct SE3Pose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * Exception to throw if an invalid orientation is given.
   */
  class InvalidOrientation : public std::exception {
    virtual const char *what() const throw() {
      return "Invalid Orientation. Not in SO(3)!";
    }
  } InvalidOrientationCreated;

  Vector3d position{0, 0, 0};
  Matrix3d orientation{Matrix3d::Identity()};

  /**
   * Default Ctor.
   */
  SE3Pose() {}

  /**
   * Constructs an SE3Pose from a position and orientation matrix.
   * @param position The position.
   * @param orientation The orientation as a matrix.
   */
  SE3Pose(const Vector3d &position, const Matrix3d &orientation) : position(position), orientation(orientation) {
    // Ensure valid orientation.
    double eps = 1e-3; // Tolerance for orientation.
    if (std::abs(orientation.determinant() - 1.0) > eps)
      throw InvalidOrientationCreated;
  }

  /**
   * Constructs an SE3Pose from a position and quaternion.
   * @param position The position.
   * @param quaternion The orientation as a quaternion.
   */
  SE3Pose(const Vector3d &position, const Vector4d &quaternion) : position(position) {
    quat2rot(quaternion, orientation); // Convert quaternion to orientation.
  }

  /**
   * @brief Construct an SE3Pose from an SE2Pose.
   * @param se2_pose The SE2Pose to construct the SE3Pose with. (X, Y, Yaw).
   */
  SE3Pose(const Vector3d &se2_pose) : position(Vector3d (se2_pose[0], se2_pose[1], 0.0)), orientation(nx::rotz(se2_pose[2]))  {}

  /**
   * Returns the Yaw of the orientation.
   * @return The yaw.
   */
  double getYaw() const {
    return std::atan2(orientation(1, 0), orientation(0, 0));
  }

  /**
   * Returns the Pitch of the orientation.
   * @return The pitch.
   */
  double getPitch() const {
    return std::atan2(-orientation(2, 0), std::hypot(orientation(2, 1), orientation(2, 2)));
  }

  /**
   * Returns the Roll of the orientation.
   * @return The roll.
   */
  double getRoll() const {
    return std::atan2(orientation(2, 1), orientation(2, 2));
  }

  /**
   * Returns the SE(2) projection of the full SE(3) Pose.
   * @return The SE(2) pose.
   */
  Vector3d getSE2() const {
    Vector3d result;
    result << position[0], position[1], getYaw();
    return result;
  }

  // TODO Implement rotation matrix to Quaternion.
//
//        Vector4d getQuaternion() const {
//
//        }
};
} // End namespace
#endif //ENVIRONMENT_SE3STATE_H
