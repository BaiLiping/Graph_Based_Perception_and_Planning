#ifndef INFO_GATHERING_RANGEMODEL_H
#define INFO_GATHERING_RANGEMODEL_H

#include <Eigen/Dense>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <random>
#include <fstream>   // std::ifstream

#include <igl/sensing/sensor.h>
#include <igl/se3_pose.h>

namespace nx {
class RangeSensor : public Sensor<SE3Pose> {
  const double min_range_;
  const double max_range_;
  const double min_hang_;
  const double max_hang_;
  const double min_vang_;
  const double max_vang_;
  const double noise_stdev_;

  // Noise generation
  std::random_device rd_;
  std::default_random_engine gen_;
  std::normal_distribution<double> dis_;

  typedef CGAL::Simple_cartesian<double> K;
  typedef K::Point_3 Point;
  typedef K::Segment_3 Segment;
  typedef CGAL::Surface_mesh<Point> Mesh;
  typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
  typedef CGAL::AABB_traits<K, Primitive> Traits;
  typedef CGAL::AABB_tree<Traits> Tree;
  Mesh mesh_;
  Tree tree_;
  bool mapSet_;

 public:
  RangeSensor(double min_range, double max_range,
              double min_hang, double max_hang,
              double min_vang, double max_vang,
              double noise_stdev, std::string off_file = "")
      : Sensor(1),
        min_range_(min_range), max_range_(max_range),
        min_hang_(min_hang), max_hang_(max_hang),
        min_vang_(min_vang), max_vang_(max_vang),
        noise_stdev_(noise_stdev), gen_(rd_()), dis_(0.0, noise_stdev_),
        mapSet_(off_file.length() > 0) {
    if (mapSet_)
      setMap(off_file);
  }

  void setMap(const std::string &off_file_name) {
    std::cout << "Setting OFF FILE\n";
    std::ifstream input(off_file_name);
    mesh_.clear();
    input >> mesh_;
    if (mesh_.number_of_faces() > 0) {
      tree_.rebuild(faces(mesh_).first, faces(mesh_).second, mesh_);
      tree_.build();
      tree_.accelerate_distance_queries();
      mapSet_ = true;
    }
  }

  Measurement sense(const SE3Pose &x, const TargetModel &tmm, int debug = 0) override {
    // Get current robot and target positions
    VectorXd y_true = tmm.GetTargetState();
    VectorXd p = x.position;
    Eigen::Matrix3d Rz = rotz(x.getYaw());

    int num_targets = tmm.num_targets();
    int y_dim = tmm.target_dim / num_targets;

    int z_dim = 1; // Range Only for each target.
    VectorXd z(num_targets * z_dim);
    VectorXd zWithNoise(num_targets * z_dim);
    VectorXd innovation(num_targets * z_dim);
    std::vector<int> da_vec;

    // Calculate Range and Bearing measurement.
    int index = 0;
    for (const auto &pair : tmm.targets)
    {
      int ID = pair.first;
      VectorXd target = pair.second->state;
      // Compute innovation separately due to bearing angle wraparound.
      Eigen::Vector2d obs_det = sense(Rz, p, target);
      z(index, 0) = obs_det(0);
      da_vec.push_back(ID);
      // Deal with range and Bearing noise.
      zWithNoise(index, 0) = std::max(std::min(z(index, 0) + dis_(gen_), max_range_), min_range_);
      innovation(index, 0) = z(index, 0) - zWithNoise(index, 0);
      index ++;
    }

    // Generate the measurement.
    Measurement m;
    m.z = z;
    m.z_dim = (unsigned) z_dim;
    m.zWithNoise = zWithNoise;
    m.innovation = innovation;
    m.da = da_vec;
    return m;
  }

//  Eigen::MatrixXd sense(const SE3Pose &x,
//                        const TargetModel &tmm) const {
//    Eigen::Vector3d p = x.position;
//    Eigen::Matrix3d Rz = rotz(x.getYaw());
//
//    Eigen::VectorXd y_true = tmm.GetTargetState();
//    int num_targets = tmm.num_targets();
//    int y_dim = tmm.target_dim / num_targets;
//
//    // Measurements generated are a 3-tuple (range, data association, detectable).
//    Eigen::MatrixXd z(num_targets, 3);
//    std::map<int, int> da = tmm.mgr.da_reverse; // This accepts lookup by target index rather than ID
//    for (int i = 0; i < num_targets; ++i) {
//      Eigen::Vector2d obs_det = sense(Rz, p, y_true.segment(y_dim * i, 3));
//      z(i, 0) = obs_det(0);
//      z(i, 1) = da[i];
//      z(i, 2) = obs_det(1);
//    }
//    return z;
//  }

  void getJacobian(Matrix<double, Dynamic, Dynamic> &H, MatrixXd &V, const SE3Pose &x, const VectorXd &y,
                   const int &y_dim) const override {

    // Get position and rotation matrices.
    Eigen::Vector3d p = x.position;
    Eigen::Matrix3d Rz = rotz(x.getYaw());

    V.setZero();
    H.setZero();
    int num_targets_known = static_cast<int>(y.rows()) / y_dim;
    for (int i = 0; i < num_targets_known; ++i) {

      Eigen::Vector3d Hi = getJacobianY(p, y.segment(y_dim * i, 3));
      if (isValid(Rz, p, y.segment(y_dim * i, 3))) {
        H.block(i, y_dim * i, 1, 3) = Hi.transpose();
      }
      double range = (p - y.segment(y_dim * i, 3)).norm();
      V(i, i) = (.1 + range / max_range_) * noise_stdev_ * noise_stdev_;
    }
  }

  // pair of (range, detectable)
  Eigen::Vector2d sense(const Eigen::Matrix3d &R,
                        const Eigen::Vector3d &p,
                        const Eigen::Vector3d &y) const {
    Eigen::Vector2d obs_det; // pair of (range, detectable)
    obs_det(1) = 0.0;
    if (!isValid(R, p, y))
      return obs_det;
    if (mapSet_) {
      Segment segment_query(Point(p.x(), p.y(), p.z()), Point(y.x(), y.y(), y.z()));
      if (tree_.do_intersect(segment_query))
        return obs_det;
    }
    obs_det(1) = 1.0;
    obs_det(0) = (p - y).norm();
    return obs_det;
  }

  bool isValid(const Eigen::Matrix3d &R,
               const Eigen::Vector3d &p,
               const Eigen::Vector3d &y) const {
    // Check if hang and vang are within bounds
    Eigen::Vector3d y_laser_frame = R.transpose() * (y - p); // world to lidar frame
    double azimuth = std::atan2(y_laser_frame.y(), y_laser_frame.x());
    if (azimuth <= min_hang_ || azimuth >= max_hang_)
      return false;
    double elevation = std::atan2(y_laser_frame.z(), std::hypot(y_laser_frame.x(), y_laser_frame.y()));
    if (elevation <= min_vang_ || elevation >= max_vang_)
      return false;
    // Check if the range is within bounds
    double d = (p - y).norm();
    if (d <= min_range_ || d >= max_range_)
      return false;
    if (mapSet_) {
      Segment segment_query(Point(p.x(), p.y(), p.z()), Point(y.x(), y.y(), y.z()));
      if (tree_.do_intersect(segment_query))
        return false;
    }
    return true;
  }

  Eigen::Vector3d getJacobianY(const Eigen::Vector3d &p,
                               const Eigen::Vector3d &y) const {
    Eigen::Vector3d H = y - p;
    // adjust for really small values
    for (int k = 0; k < 3; ++k)
      if (std::abs(H(k)) < 0.00001)
        H(k) = ((0.0 <= H(k)) - (H(k) < 0.0)) * 0.00001;
    H = H / H.norm();
    return H;
  }
};
}

#endif //INFO_GATHERING_RANGEMODEL_H
