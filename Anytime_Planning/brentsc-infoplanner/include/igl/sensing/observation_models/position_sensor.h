//
// Created by brent on 2/20/19.
//

#ifndef INFORMATION_GATHERING_LIBRARY_POSITION_SENSOR_H
#define INFORMATION_GATHERING_LIBRARY_POSITION_SENSOR_H

#include <random>
#include <Eigen/Dense>
#include <map>
#include <igl/sensing/sensor.h>
#include <igl/robot.h>
#include <igl/se3_pose.h>

using namespace Eigen;
using Vector1d = Matrix<double, 1, 1>;

namespace nx {

class PositionSensor : public nx::Sensor<SE3Pose> {

 public:
  // 3-D Position Sensor Parameters
  const double min_range_;
  const double max_range_;
  const double min_hang_;
  const double max_hang_;
  const double min_vang_;
  const double max_vang_;
  // Noise
  const double p_sigma_; // Standard deviation of position sensor noise.

  // Map and CMap for ray-tracing.
  std::shared_ptr<nx::map_nd> map;
  const std::vector<char> cmap;

  /**
   * @brief Constructs a PositionSensor.
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
  PositionSensor(double min_range,
              double max_range,
              double min_hang,
              double max_hang,
              double min_vang,
              double max_vang,
              double p_sigma,
              std::shared_ptr<nx::map_nd> map,
              const std::vector<char> &cmap) : Sensor(2),
                                               min_range_(min_range),
                                               max_range_(max_range),
                                               min_hang_(M_PI / 180 * min_hang),
                                               max_hang_(M_PI / 180 * max_hang),
                                               min_vang_(M_PI / 180 * min_vang),
                                               max_vang_(M_PI / 180 * max_vang),
                                               p_sigma_(p_sigma),
                                               map(map), cmap(cmap) {}

  /**
   * Computes the observation model h(x,y) as a function of a robot state x and a single target state y.
   * @param x The sensor state x.
   * @param y The target state y.
   * @return The evaluated observation model h(x,y).
   */
  VectorXd observationModel(const SE3Pose &x, const VectorXd &y) const override {
    Vector2d z;
    z << y[0], y[1];
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
    Vector2d z = observationModel(x, y);
    z[0] += nx::normal_dist(0, p_sigma_ * p_sigma_);
    z[1] += nx::normal_dist(0, p_sigma_ * p_sigma_);
    Measurement m(z, target.ID, isValid(x.orientation, x.position, y));
    return m;
  }

  void getJacobian(MatrixXd &H, MatrixXd &V, const SE3Pose &x, const VectorXd &y, bool check_valid) const override {
    V.setZero(); // Reset Jacobian Matrices
    H.setZero();
    Vector3d y_;
    y_ << y[0], y[1], 0;
    if (!check_valid || isValid(x.orientation, x.position, y_)) {
      H(0, 0) = 1;
      H(1, 1) = 1;
    }
    V(0, 0) = p_sigma_ * p_sigma_;
    V(1, 1) = p_sigma_ * p_sigma_;
  }

  MatrixXd maxSensorMatrix(const SE3Pose &x, const VectorXd &y, const std::shared_ptr<TargetModel> &tmm, int T) const override {
    MatrixXd M = MatrixXd::Zero(y.rows(), y.rows());
    int index = 0;
    for (const auto &target: tmm->targets)
    {
      int y_dim = target.second->y_dim;
      Vector2d z = observationModel(x, y.segment(index * y_dim, y_dim));
      double dist = std::hypot(z[0], z[1]);
      if (dist< 1.5*T + max_range_)     // Check if the target is reachable in T timesteps.
      {
        M.block(index * y_dim, index * y_dim, y_dim, y_dim) = 1.0 / (p_sigma_ * p_sigma_) * MatrixXd::Identity(y_dim, y_dim);
      }
      index++;
    }
    return M;
  }

 private:


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

}; // End class
} // End namespace

#endif //INFORMATION_GATHERING_LIBRARY_POSITION_SENSOR_H
