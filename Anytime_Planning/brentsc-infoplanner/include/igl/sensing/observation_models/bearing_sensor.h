//
// Created by brent on 2/16/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_BEARING_H
#define INFORMATION_GATHERING_LIBRARY_BEARING_H

#include <random>
#include <Eigen/Dense>
#include <map>
#include <igl/sensing/sensor.h>
#include <igl/robot.h>
#include <igl/se3_pose.h>

using namespace Eigen;
using Vector1d = Matrix<double, 1, 1>;

namespace nx {

/**
 * @brief This class implements a Bearing sensor, capable of simulation and modeling a relative orientation between two
 * planar points.
 */
class BearingSensor : public nx::Sensor<SE3Pose> {
 public:
  // Bearing Sensor Parameters
  const double min_range_;
  const double max_range_;
  const double min_hang_;
  const double max_hang_;
  double fov;
  const double b_sigma_; // Bearing Noise covariance.

  // Map and CMap for ray-tracing.
  std::shared_ptr<nx::map_nd> map;
  const std::vector<char> cmap;

  /**
   * Constructs a BearingSensor.
   * @param min_range Minimum sensing range.
   * @param max_range Maximum sensing range.
   * @param min_hang Minimum horizontal angle (degrees).
   * @param max_hang Maximum horizontal angle (degrees)
   * @param min_vang Minimum vertical angle (degrees).
   * @param max_vang Maximum vertical angle (degrees)
   * @param r_sigma Standard deviation of sensing noise.
   * @param map Map parameters for bounds checking.
   * @param cmap Costmap for ray-tracing.
   */
  BearingSensor(double min_range,
                double max_range,
                double min_hang,
                double max_hang,
                double b_sigma,
                std::shared_ptr<nx::map_nd> map,
                const std::vector<char> &cmap) : Sensor(1),
                                                 min_range_(min_range),
                                                 max_range_(max_range),
                                                 min_hang_(M_PI / 180 * min_hang),
                                                 max_hang_(M_PI / 180 * max_hang),
                                                 b_sigma_(b_sigma),
                                                 fov(max_hang_ - min_hang_),
                                                 map(map), cmap(cmap) {}

  /**
   * Computes the observation model h(x,y) as a function of a robot state x and a single target state y.
   * @param x The sensor state x.
   * @param y The target state y.
   * @return The evaluated observation model h(x,y).
   */
  VectorXd observationModel(const SE3Pose &x, const VectorXd &y) const override {
    Vector1d z;
    z << computeBearing(x.position.head(2), y.head(2), x.getYaw());
    return z;
  }

  /**
   * Computes a single noisy measurement vector of the target from a robot state x.
   * @param x The state sensing from.
   * @param target The target being sensed.
   * @return The resulting measurement.
   */
  Measurement sense(const SE3Pose &x, const Target &target) override {
    Vector3d y = target.getPosition();
    Vector1d z = observationModel(x, y);
    z[0] += nx::normal_dist(0, b_sigma_ * b_sigma_);//(1+ std::abs(z[0] * 180 / 3.14159 / 120)));
    Measurement m(z, target.ID, isValid(x.orientation, x.position, y));
    return m;
  }

  void getJacobian(MatrixXd &H, MatrixXd &V, const SE3Pose &x, const VectorXd &y, bool check_valid) const override {
    V.setZero(); // Reset Jacobian Matrices
    H.setZero();
    Vector1d z = observationModel(x, y); // z = (bearing)
    double range = (x.position.head(2) - y.head(2)).norm() + 0.001;
    Vector3d target_position;
    target_position << y[0], y[1], 0;
    if (!check_valid || isValid(x.orientation, x.position, target_position)) {
      H(0, 0) = -(y[1] - x.position[1]) / (range * range);
      H(0, 1) = (y[0] - x.position[0]) / (range * range);
    }
    V(0, 0) = b_sigma_ * b_sigma_;// * (1+ std::abs(z[0] * 180 / 3.14159 / 120)); // fov=120;
  }

  MatrixXd maxSensorMatrix(const SE3Pose &x, const VectorXd &y, const std::shared_ptr<TargetModel> &tmm, int T) const override {
    MatrixXd M = MatrixXd::Zero(y.rows(), y.rows());
    int index = 0;
    for (const auto &target: tmm->targets)
    {
      int y_dim = target.second->y_dim;
      double range = std::hypot(x.position[0] - target.second->getPosition()[0],
          x.position[1] - target.second->getPosition()[1]);
      if (range < 1.5*T + max_range_)     // Check if the target is reachable in T timesteps.
      {
        double dist = range < 1.5 * T ? 0.0 : range - 1.5 * T;
        M.block(index * y_dim, index * y_dim, y_dim, y_dim) = 1.0 /
            (b_sigma_ * b_sigma_ * std::max(min_range_, dist)) * MatrixXd::Identity(y_dim, y_dim);
      }
      index++;
    }
    return M;
  }
  /**
   * Compute innovation between the measurement and observation models accounting for angle wrap-around.
   * @param measurement The measurement generated.
   * @param predicted_measurement The measurement predicted by an observation model.
   * @return The innovation between the two vectors.
   */
  virtual VectorXd computeInnovation(const VectorXd &measurement,
                                     const VectorXd &predicted_measurement) const override {
    VectorXd innovation = measurement - predicted_measurement;
    for (int i = 0; i < innovation.rows(); i++)
      innovation[i] = nx::restrict_angle(innovation[i]);
    return innovation;
  };

 private:

/**
 * Computes the bearing between two agent positions, where the first agent has an associated yaw.
 * @param x The robot position.
 * @param y The target position.
 * @param yaw The heading of the .
 * @return The resulting bearing angle in radians.
 */
  double computeBearing(const VectorXd &x, const VectorXd &y, double yaw) const {
    return nx::restrict_angle(std::atan2((y[1] - x[1]),
                                         (y[0] - x[0])) - yaw); // Bearing
  }

  /**
   * @brief Checks whether a given sensing configuration may return a valid sensor measurement.
   * @param R The orientation of the sensor with respect to the global fixed frame.
   * @param p The position of the sensor in the global fixed frame.
   * @param y The position of the target in the global fixed frame.
   * @return True if the sensing configuration is valid.
   */
  bool isValid(const Eigen::Matrix3d &R,
               const Eigen::Vector3d &p,
               const Eigen::Vector3d &y) const {
    // Check if hang and vang are within bounds
    Eigen::Vector3d y_laser_frame = R.transpose() * (y - p); // world to lidar frame
    double azimuth = std::atan2(y_laser_frame.y(), y_laser_frame.x());
    if (azimuth <= min_hang_ || azimuth >= max_hang_)
      return false;
    // Check if the bearing is within bounds
    double d = (p - y).norm();
    if (d <= min_range_ || d >= max_range_)
      return false;
    bool collision = false;
    std::vector<int> x_coord;// The X and Y Coordinates of the raster
    std::vector<int> y_coord;
    double sx = p[0];
    double sy = p[1];
    double ex = y[0];
    double ey = y[1];
    bresenham2D(sx, sy, ex, ey, map->min()[0], map->min()[1], map->res()[0], map->res()[1], x_coord, y_coord);
    for (int i = 0; i < x_coord.size(); i++) {
      unsigned int lindix = map->subv2ind_colmajor({x_coord[i], y_coord[i]});
      if (lindix < 0 || lindix > cmap.size()) { // Cell is out of bounds, continue to the next coordinate.
        continue;
      } else if (cmap.at(map->subv2ind_colmajor({x_coord[i], y_coord[i]})) == '1') {
        collision = true;
        break;
      }
    }
    return !collision;
  }
};
}
#endif //INFORMATION_GATHERING_LIBRARY_BEARING_H
