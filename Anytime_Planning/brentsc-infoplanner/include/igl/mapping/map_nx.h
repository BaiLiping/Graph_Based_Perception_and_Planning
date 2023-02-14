#ifndef __NX_MAP_UTILS_H_
#define __NX_MAP_UTILS_H_
#include <cmath>
#include <vector>
#include <limits>
#include <Eigen/Core>
//#include <iostream> // for debugging

// for saving the map
#include <fstream>               // std::ofstream
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>  // boost::filesystem::path
#include <typeinfo>

namespace nx {
inline int meters2cells(double datam, double min, double res) {
  return static_cast<int>(std::floor((datam - min) / res));
}
inline double cells2meters(int datac, double min, double res) {
  return (static_cast<double>(datac) + 0.5) * res + min;
}
inline double meters2cells_cont(double datam, double dim_min, double res) {
  return (datam - dim_min) / res - 0.5;
}

// returns the first odd integer larger than x
inline int odd_ceil(double x) {
  if (std::abs(std::floor(x) - x) <= std::numeric_limits<double>::epsilon())
    x = std::floor(x);
  int ocx = static_cast<int>(std::ceil(x));
  return (ocx % 2 == 0) ? (ocx + 1) : (ocx);
}

// Row major order as in C++
inline int subv2ind_rowmajor(std::vector<int>::const_iterator const &datac_begin,
                             std::vector<int>::const_iterator const &datac_end,
                             std::vector<int>::const_iterator const &size_begin,
                             std::vector<int>::const_iterator const &size_end) {
  if (datac_end <= datac_begin || size_end <= size_begin) return -1;
  int idx = *(datac_end - 1);
  int prod = 1;
  std::vector<int>::const_iterator it1 = datac_end - 2;
  std::vector<int>::const_iterator it2 = size_end - 1;
  for (; it1 != (datac_begin - 1) && it2 != size_begin; --it1, --it2) {
    prod *= (*it2);
    idx += prod * (*it1);
  }
  return idx;
}

inline std::vector<int> ind2subv_rowmajor(int ind,
                                          const std::vector<int>::const_iterator &size_begin,
                                          const std::vector<int>::const_iterator &size_end) {
  const size_t ndims = std::distance(size_begin, size_end);
  std::vector<int> subv(ndims);
  std::vector<int>::const_iterator it = size_end - 1;
  for (int k = ndims - 1; k >= 0; --k, --it) {
    subv[k] = ind % (*it);
    ind -= subv[k];
    ind /= (*it);
  }
  return subv;
}

// Column major order as in MATLAB
inline int subv2ind_colmajor(std::vector<int>::const_iterator const &datac_begin,
                             std::vector<int>::const_iterator const &datac_end,
                             std::vector<int>::const_iterator const &size_begin,
                             std::vector<int>::const_iterator const &size_end) {
  if (datac_end <= datac_begin || size_end <= size_begin) return -1;
  int idx = *datac_begin;
  int prod = 1;
  std::vector<int>::const_iterator it1 = datac_begin + 1;
  std::vector<int>::const_iterator it2 = size_begin;
  for (; it1 != datac_end && it2 != (size_end - 1); ++it1, ++it2) {
    prod *= (*it2);
    idx += (*it1) * prod;
  }
  return idx;
}

inline std::vector<int> ind2subv_colmajor(int ind,
                                          const std::vector<int>::const_iterator &size_begin,
                                          const std::vector<int>::const_iterator &size_end) {
  const size_t ndims = std::distance(size_begin, size_end);
  std::vector<int> subv(ndims);
  std::vector<int>::const_iterator it = size_begin;
  for (size_t k = 0; k < ndims; ++k, ++it) {
    subv[k] = ind % (*it);
    ind -= subv[k];
    ind /= (*it);
  }
  return subv;
}


/******************************************************************************/
// Vector Implementations
inline std::vector<int> meters2cells(std::vector<double>::const_iterator const &datam_begin,
                                     std::vector<double>::const_iterator const &datam_end,
                                     std::vector<double>::const_iterator const &dim_min_begin,
                                     std::vector<double>::const_iterator const &dim_min_end,
                                     std::vector<double>::const_iterator const &res_begin,
                                     std::vector<double>::const_iterator const &res_end) {
  std::vector<int> datac;
  std::vector<double>::const_iterator it1 = datam_begin;
  std::vector<double>::const_iterator it2 = dim_min_begin;
  std::vector<double>::const_iterator it3 = res_begin;
  for (; it1 != datam_end; ++it1, ++it2, ++it3)
    datac.push_back(meters2cells(*it1, *it2, *it3));
  return datac;
}

inline std::vector<double> cells2meters(std::vector<int>::const_iterator const &datac_begin,
                                        std::vector<int>::const_iterator const &datac_end,
                                        std::vector<double>::const_iterator const &dim_min_begin,
                                        std::vector<double>::const_iterator const &dim_min_end,
                                        std::vector<double>::const_iterator const &res_begin,
                                        std::vector<double>::const_iterator const &res_end) {
  std::vector<double> datam;
  std::vector<int>::const_iterator it1 = datac_begin;
  std::vector<double>::const_iterator it2 = dim_min_begin;
  std::vector<double>::const_iterator it3 = res_begin;
  for (; it1 != datac_end; ++it1, ++it2, ++it3)
    datam.push_back(meters2cells(*it1, *it2, *it3));
  return datam;
}

inline std::vector<double> meters2cells_cont(
    std::vector<double>::const_iterator const &datam_begin,
    std::vector<double>::const_iterator const &datam_end,
    std::vector<double>::const_iterator const &dim_min_begin,
    std::vector<double>::const_iterator const &dim_min_end,
    std::vector<double>::const_iterator const &res_begin,
    std::vector<double>::const_iterator const &res_end) {
  std::vector<double> datac;
  std::vector<double>::const_iterator it1 = datam_begin;
  std::vector<double>::const_iterator it2 = dim_min_begin;
  std::vector<double>::const_iterator it3 = res_begin;
  for (; it1 != datam_end; ++it1, ++it2, ++it3)
    datac.push_back(meters2cells_cont(*it1, *it2, *it3));
  return datac;
}

inline std::vector<int> meters2cells(std::vector<double> const &datam,
                                     std::vector<double> const &dim_min,
                                     std::vector<double> const &res) {
  std::vector<int> datac(datam.size());
  for (unsigned k = 0; k < datam.size(); ++k)
    datac[k] = meters2cells(datam[k], dim_min[k], res[k]);
  return datac;
}

inline std::vector<int> meters2cells(std::vector<double> const &datam,
                                     std::vector<double> const &dim_min, double res) {
  std::vector<int> datac(datam.size());
  for (unsigned k = 0; k < datam.size(); ++k)
    datac[k] = meters2cells(datam[k], dim_min[k], res);
  return datac;
}

inline std::vector<double> cells2meters(std::vector<int> const &datac,
                                        std::vector<double> const &dim_min, std::vector<double> const &res) {
  std::vector<double> datam(datac.size());
  for (unsigned k = 0; k < datac.size(); ++k)
    datam[k] = cells2meters(datac[k], dim_min[k], res[k]);
  return datam;
}

inline std::vector<double> cells2meters(std::vector<int> const &datac,
                                        std::vector<double> const &dim_min, double res) {
  std::vector<double> datam(datac.size());
  for (unsigned k = 0; k < datac.size(); ++k)
    datam[k] = cells2meters(datac[k], dim_min[k], res);
  return datam;
}

inline std::vector<double> meters2cells_cont(const std::vector<double> &datam,
                                             const std::vector<double> &dim_min,
                                             const std::vector<double> &res) {
  std::vector<double> datac(datam.size());
  for (unsigned k = 0; k < datam.size(); ++k)
    datac[k] = meters2cells_cont(datam[k], dim_min[k], res[k]);
  return datac;
}

inline std::vector<double> meters2cells_cont(const std::vector<double> &datam,
                                             const std::vector<double> &dim_min,
                                             double res) {
  std::vector<double> datac(datam.size());
  for (unsigned k = 0; k < datam.size(); ++k)
    datac[k] = meters2cells_cont(datam[k], dim_min[k], res);
  return datac;
}

/******************************************************************************/
// Matrix Implementations
inline Eigen::MatrixXi meters2cells(const Eigen::MatrixXd &datam,
                                    const Eigen::VectorXd &dim_min,
                                    const Eigen::VectorXd &res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXi datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells(datam(d, p), dim_min(d), res(d));
  return datac;
}

inline Eigen::MatrixXi meters2cells(const Eigen::MatrixXd &datam,
                                    const Eigen::VectorXd &dim_min,
                                    double res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXi datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells(datam(d, p), dim_min(d), res);
  return datac;
}

inline Eigen::MatrixXd cells2meters(const Eigen::MatrixXi &datac,
                                    const Eigen::VectorXd &dim_min,
                                    const Eigen::VectorXd &res) {
  // datac = [num_dim x num_pts]
  const size_t num_dim = datac.rows();
  const size_t num_pts = datac.cols();
  Eigen::MatrixXd datam(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datam(d, p) = cells2meters(datac(d, p), dim_min(d), res(d));
  return datam;
}

inline Eigen::MatrixXd cells2meters(const Eigen::MatrixXi &datac,
                                    const Eigen::VectorXd &dim_min,
                                    double res) {
  // datac = [num_dim x num_pts]
  const size_t num_dim = datac.rows();
  const size_t num_pts = datac.cols();
  Eigen::MatrixXd datam(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datam(d, p) = cells2meters(datac(d, p), dim_min(d), res);
  return datam;
}

inline Eigen::MatrixXd meters2cells_cont(const Eigen::MatrixXd &datam,
                                         const Eigen::VectorXd &dim_min,
                                         const Eigen::VectorXd &res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXd datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells_cont(datam(d, p), dim_min(d), res(d));
  return datac;
}

inline Eigen::MatrixXd meters2cells_cont(const Eigen::MatrixXd &datam,
                                         const Eigen::VectorXd &dim_min,
                                         double res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXd datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells_cont(datam(d, p), dim_min(d), res);
  return datac;
}

inline Eigen::MatrixXi meters2cells(const Eigen::MatrixXd &datam,
                                    const std::vector<double> &dim_min,
                                    const std::vector<double> &res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXi datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells(datam(d, p), dim_min[d], res[d]);
  return datac;
}

inline Eigen::MatrixXi meters2cells(const Eigen::MatrixXd &datam,
                                    const std::vector<double> &dim_min,
                                    double res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXi datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells(datam(d, p), dim_min[d], res);
  return datac;
}

inline Eigen::MatrixXd cells2meters(const Eigen::MatrixXi &datac,
                                    const std::vector<double> &dim_min,
                                    const std::vector<double> &res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datac.rows();
  const size_t num_pts = datac.cols();
  Eigen::MatrixXd datam(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datam(d, p) = cells2meters(datac(d, p), dim_min[d], res[d]);
  return datam;
}

inline Eigen::MatrixXd cells2meters(const Eigen::MatrixXi &datac,
                                    const std::vector<double> &dim_min,
                                    double res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datac.rows();
  const size_t num_pts = datac.cols();
  Eigen::MatrixXd datam(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datam(d, p) = cells2meters(datac(d, p), dim_min[d], res);
  return datam;
}

inline Eigen::MatrixXd meters2cells_cont(const Eigen::MatrixXd &datam,
                                         const std::vector<double> &dim_min,
                                         const std::vector<double> &res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXd datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells_cont(datam(d, p), dim_min[d], res[d]);
  return datac;
}

inline Eigen::MatrixXd meters2cells_cont(const Eigen::MatrixXd &datam,
                                         const std::vector<double> &dim_min,
                                         double res) {
  // datam = [num_dim x num_pts]
  const size_t num_dim = datam.rows();
  const size_t num_pts = datam.cols();
  Eigen::MatrixXd datac(num_dim, num_pts);
  for (unsigned d = 0; d < num_dim; ++d)
    for (unsigned p = 0; p < num_pts; ++p)
      datac(d, p) = meters2cells_cont(datam(d, p), dim_min[d], res);
  return datac;
}
/******************************************************************************/

class map_nd {
  std::vector<double> min_;
  std::vector<double> max_;
  std::vector<double> res_;
  std::vector<int> size_;
  std::vector<double> origin_;
  std::vector<int> origincells_;

 public:
  map_nd() {} // default constructor

  map_nd(std::vector<double> const &devmin, std::vector<double> const &devmax,
         std::vector<double> const &resolution)
      : res_(resolution) {
    double dev;
    for (unsigned k = 0; k < devmin.size(); ++k) {
      origin_.push_back((devmin[k] + devmax[k]) / 2);
      size_.push_back(nx::odd_ceil((devmax[k] - devmin[k]) / resolution[k]));
      dev = size_[k] * resolution[k] / 2;
      min_.push_back(origin_[k] - dev);
      max_.push_back(origin_[k] + dev);
      origincells_.push_back((size_[k] - 1) / 2);
    }
  }

  map_nd(std::vector<double> const &devmin, std::vector<double> const &devmax,
         double resolution)
      : res_(devmin.size(), resolution) {
    double dev;
    for (unsigned k = 0; k < devmin.size(); ++k) {
      origin_.push_back((devmin[k] + devmax[k]) / 2);
      size_.push_back(nx::odd_ceil((devmax[k] - devmin[k]) / resolution));
      dev = size_[k] * resolution / 2;
      min_.push_back(origin_[k] - dev);
      max_.push_back(origin_[k] + dev);
      origincells_.push_back((size_[k] - 1) / 2);
    }
  }

  map_nd(size_t num_dim, double *devmin, double *devmax, double *resolution)
      : res_(resolution, resolution + num_dim) {
    double dev;
    for (unsigned k = 0; k < num_dim; ++k) {
      origin_.push_back((devmin[k] + devmax[k]) / 2);
      size_.push_back(nx::odd_ceil((devmax[k] - devmin[k]) / resolution[k]));
      dev = size_[k] * resolution[k] / 2;
      min_.push_back(origin_[k] - dev);
      max_.push_back(origin_[k] + dev);
      origincells_.push_back((size_[k] - 1) / 2);
    }
  }

  map_nd(size_t num_dim, double *devmin, double *devmax, double resolution)
      : res_(num_dim, resolution) {
    double dev;
    for (unsigned k = 0; k < num_dim; ++k) {
      origin_.push_back((devmin[k] + devmax[k]) / 2);
      size_.push_back(nx::odd_ceil((devmax[k] - devmin[k]) / resolution));
      dev = size_[k] * resolution / 2;
      min_.push_back(origin_[k] - dev);
      max_.push_back(origin_[k] + dev);
      origincells_.push_back((size_[k] - 1) / 2);
    }
  }

  map_nd(const std::vector<int> &oddsize, const std::vector<double> &origin,
         const std::vector<double> &resolution)
      : res_(resolution), size_(oddsize), origin_(origin) {
    double dev;
    for (unsigned k = 0; k < size_.size(); ++k) {
      if (size_[k] % 2 == 0)
        size_[k] += 1; // ensure the map size is odd!
      dev = size_[k] * res_[k] / 2;
      min_.push_back(origin_[k] - dev);
      max_.push_back(origin_[k] + dev);
      origincells_.push_back((size_[k] - 1) / 2);
    }
  }

  // Methods
  std::vector<double> const &min(void) const { return min_; }
  std::vector<double> const &max(void) const { return max_; }
  std::vector<double> const &res(void) const { return res_; }
  std::vector<int> const &size(void) const { return size_; }
  std::vector<double> const &origin(void) const { return origin_; }
  std::vector<int> const &origincells(void) const { return origincells_; }

  std::vector<int> meters2cells(const std::vector<double> &datam) const { return nx::meters2cells(datam, min_, res_); }

  std::vector<double> cells2meters(const std::vector<int> &datac) const { return nx::cells2meters(datac, min_, res_); }

  std::vector<double> meters2cells_cont(const std::vector<double> &datam) const {
    return nx::meters2cells_cont(datam,
                                 min_,
                                 res_);
  }

  std::vector<int> meters2cells(const std::vector<double>::const_iterator &datam_begin,
                                const std::vector<double>::const_iterator &datam_end) const {
    return nx::meters2cells(datam_begin,
                            datam_end,
                            min_.begin(),
                            min_.end(),
                            res_.begin(),
                            res_.end());
  }

  std::vector<double> cells2meters(const std::vector<int>::const_iterator &datac_begin,
                                   const std::vector<int>::const_iterator &datac_end) const {
    return nx::cells2meters(datac_begin,
                            datac_end,
                            min_.begin(),
                            min_.end(),
                            res_.begin(),
                            res_.end());
  }

  std::vector<double> meters2cells_cont(const std::vector<double>::const_iterator &datam_begin,
                                        const std::vector<double>::const_iterator &datam_end) const {
    return nx::meters2cells_cont(datam_begin,
                                 datam_end,
                                 min_.begin(),
                                 min_.end(),
                                 res_.begin(),
                                 res_.end());
  }

  // Row major order as in C++
  int subv2ind_rowmajor(std::vector<int> const &datac) const {
    const size_t ndims = datac.size();
    if (ndims == 0) return -1;

    int idx = datac.back();
    int prod = 1;
    for (int k = ndims - 1; k > 0; --k) {
      prod = prod * size_[k];
      idx += datac[k - 1] * prod;
    }
    return idx;
  }

  std::vector<int> ind2subv_rowmajor(int ind) const {
    const size_t ndims = size_.size();
    std::vector<int> subv(ndims);
    for (int k = ndims - 1; k >= 0; --k) {
      subv[k] = ind % size_[k];
      ind -= subv[k];
      ind /= size_[k];
    }
    return subv;
  }

  // Column major order as in MATLAB
  int subv2ind_colmajor(std::vector<int> const &datac) const {
    const size_t ndims = datac.size();
    if (ndims == 0) return -1;

    int idx = datac[0];
    int prod = 1;
    for (size_t k = 1; k < ndims; ++k) {
      prod *= size_[k - 1];
      idx += datac[k] * prod;
    }
    return idx;
  }

  std::vector<int> ind2subv_colmajor(int ind) const {
    const size_t ndims = size_.size();
    std::vector<int> subv(ndims);
    for (size_t k = 0; k < ndims; ++k) {
      subv[k] = ind % size_[k];
      ind -= subv[k];
      ind /= size_[k];
    }
    return subv;
  }

  template<class T>
  std::vector<T> rowmajor2colmajor(const std::vector<T> &map) const {
    const size_t mapsz = map.size();
    std::vector<T> newmap(mapsz);
    for (int k = 0; k < mapsz; ++k)
      newmap[subv2ind_colmajor(ind2subv_rowmajor(k))] = map[k];
    return newmap;
  }

  template<class T>
  std::vector<T> colmajor2rowmajor(const std::vector<T> &map) const {
    const size_t mapsz = map.size();
    std::vector<T> newmap(mapsz);
    for (int k = 0; k < mapsz; ++k)
      newmap[subv2ind_rowmajor(ind2subv_colmajor(k))] = map[k];
    return newmap;
  }

  template<class T>
  void saveToYaml(const std::vector<T> &map, std::string pathToYaml) const {
    saveToYaml(map, pathToYaml, "rowmajor");
  }

  /*
   * storageorder in {rowmajor, colmajor}
   */
  template<class T>
  void saveToYaml(const std::vector<T> &map, std::string pathToYaml, std::string storageorder) const {
    // Find the file path and file name
    boost::filesystem::path bfp(pathToYaml);
    std::string yaml_parent_path(bfp.parent_path().string());
    std::string yaml_name(bfp.stem().string());

    // Generate the YAML file
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "mapmin";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < min_.size(); ++k)
      out << min_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "mapmax";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < max_.size(); ++k)
      out << max_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "mapres";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < res_.size(); ++k)
      out << res_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "mapdim";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < size_.size(); ++k)
      out << size_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "origin";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < origin_.size(); ++k)
      out << origin_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "origincells";
    out << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    for (int k = 0; k < origincells_.size(); ++k)
      out << origincells_[k];
    out << YAML::EndSeq;
    out << YAML::Key << "datatype";
    out << YAML::Value << typeid(T).name();
    out << YAML::Key << "storage";
    out << YAML::Value << storageorder;
    out << YAML::Key << "mappath";
    out << YAML::Value << yaml_name + ".cfg";
    out << YAML::EndMap;

    // Save yaml file
    std::ofstream ofs;
    ofs.open(yaml_parent_path + "/" + yaml_name + ".yaml", std::ofstream::out | std::ofstream::trunc);
    ofs << out.c_str();
    ofs.close();

    // Save cfg file
    ofs.open(yaml_parent_path + "/" + yaml_name + ".cfg", std::ofstream::out | std::ofstream::trunc);
    for (int k = 0; k < map.size(); ++k)
      ofs << map[k];
    ofs.close();
  }

  template<class T>
  void initFromYaml(std::vector<T> &map, std::string pathToYaml) {
    // Set map size
    YAML::Node map_spec = YAML::LoadFile(pathToYaml);
    if (map_spec["mapdim"] && map_spec["origin"] && map_spec["mapres"] && map_spec["mappath"]) {
      size_ = map_spec["mapdim"].as<std::vector<int>>();
      origin_ = map_spec["origin"].as<std::vector<double>>();
      res_ = map_spec["mapres"].as<std::vector<double>>();
      double dev;
      for (unsigned k = 0; k < size_.size(); ++k) {
        if (size_[k] % 2 == 0)
          size_[k] += 1; // ensure the map size is odd!
        dev = size_[k] * res_[k] / 2;
        min_.push_back(origin_[k] - dev);
        max_.push_back(origin_[k] + dev);
        origincells_.push_back((size_[k] - 1) / 2);
      }
    } else if (map_spec["mapmin"] && map_spec["mapmax"] && map_spec["mapres"] && map_spec["mappath"]) {
      const std::vector<double> &devmin = map_spec["mapmin"].as<std::vector<double>>();
      const std::vector<double> &devmax = map_spec["mapmax"].as<std::vector<double>>();
      res_ = map_spec["mapres"].as<std::vector<double>>();
      double dev;
      for (unsigned k = 0; k < devmin.size(); ++k) {
        origin_.push_back((devmin[k] + devmax[k]) / 2);
        size_.push_back(nx::odd_ceil((devmax[k] - devmin[k]) / res_[k]));
        dev = size_[k] * res_[k] / 2;
        min_.push_back(origin_[k] - dev);
        max_.push_back(origin_[k] + dev);
        origincells_.push_back((size_[k] - 1) / 2);
      }
    }

    // Read the map
    // TODO Check this
    if (size_.size() > 0) {
      std::string yaml_parent_path(boost::filesystem::path(pathToYaml).parent_path().string());
      std::ifstream fsr(yaml_parent_path + "/" + map_spec["mappath"].as<std::string>(), std::ifstream::in);
      int mapsz = 1;
      for (size_t k = 0; k < size_.size(); ++k)
        mapsz *= size_[k];
      map.resize(mapsz);
      for (int k = 0; k < mapsz; ++k)
        fsr >> map[k];
      fsr.close();
    }
  }
};

/******************************************************************************/
/******************************************************************************/
inline void bresenhamStep(double dv, double sv, int svc, double vmin, double vres,
                          int &stepV, double &tDeltaV, double &tMaxV) {
  if (dv > 0) {
    stepV = 1;
    tDeltaV = vres / dv;
    tMaxV = (vmin + (svc + 1) * vres - sv) / dv; // parametric distance until the first crossing
  } else if (dv < 0) {
    stepV = -1;
    tDeltaV = vres / -dv;
    tMaxV = (vmin + (svc) * vres - sv) / dv;
  } else {
    stepV = 0;
    tDeltaV = 0.0;
    tMaxV = std::numeric_limits<double>::infinity(); // the line doesn't cross the next plane
  }
}

inline void bresenham2D(double sx, double sy,
                        double ex, double ey,
                        double xmin, double ymin,
                        double xres, double yres,
                        std::vector<int> &xvec,
                        std::vector<int> &yvec) {
  xvec.clear();
  yvec.clear();
  // find start and end cells
  int sxc = meters2cells(sx, xmin, xres);
  int exc = meters2cells(ex, xmin, xres);
  int syc = meters2cells(sy, ymin, yres);
  int eyc = meters2cells(ey, ymin, yres);
  double dx = ex - sx;
  double dy = ey - sy;
  int stepX, stepY;          // direction of grid traversal
  double tDeltaX, tDeltaY; // parametric step size along different dimensions
  double tMaxX, tMaxY; // used to determine the dimension of the next step along the line
  bresenhamStep(dx, sx, sxc, xmin, xres, stepX, tDeltaX, tMaxX);
  bresenhamStep(dy, sy, syc, ymin, yres, stepY, tDeltaY, tMaxY);

  // Add initial voxel to the list
  xvec.push_back(sxc);
  yvec.push_back(syc);

//  std::cout <<"Sxc: " << sxc << std::endl;
//  std::cout <<"exc: " << exc << std::endl;
//  std::cout <<"Syc: " << syc << std::endl;
//  std::cout <<"Eyc: " << eyc << std::endl;
//  std::cout << "stepX "<< stepX << std::endl;
//  std::cout << "stepX "<< stepY << std::endl;
//  std::cout << "tMaxX" << tMaxX << std::endl;
//  std::cout << "tMaxY" << tMaxY << std::endl;
//  std::cout << "tDelX" << tDeltaX << std::endl;
//  std::cout << "tDelY" << tDeltaY << std::endl;
  int count = 0;
  // TODO Fix this. There is an off by one error here!! Current solution is a hack.
  while (std::abs(sxc - exc) > 1 || std::abs(syc - eyc) > 1) {

//    if (count < 40) {
//
//      std::cout <<"Sxc: " << sxc << std::endl;
//      std::cout <<"exc: " << exc << std::endl;
//      std::cout <<"Syc: " << syc << std::endl;
//      std::cout <<"Eyc: " << eyc << std::endl;
//      std::cout << "tMaxX" << tMaxX << std::endl;
//      std::cout << "tMaxY" << tMaxY << std::endl;
//      std::cout << "tDelX" << tDeltaX << std::endl;
//      std::cout << "tDelY" << tDeltaY << std::endl;
//      count++;
//    }
    if (tMaxX < tMaxY ){//- 1e-2) {
      sxc += stepX;
      tMaxX += tDeltaX;
    } else {
      syc += stepY;
      tMaxY += tDeltaY;
    }
    xvec.push_back(sxc);
    yvec.push_back(syc);
  }
}
// http://www.cse.yorku.ca/~amana/research/grid.pdf
// J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing"
// https://www.mathworks.com/matlabcentral/fileexchange/56527-fast-raytracing-through-a-3d-grid
inline void bresenham3d(double sx, double sy, double sz,
                        double ex, double ey, double ez,
                        double xmin, double ymin, double zmin,
                        double xres, double yres, double zres,
                        std::vector<int> &xvec,
                        std::vector<int> &yvec,
                        std::vector<int> &zvec) {
  xvec.clear();
  yvec.clear();
  zvec.clear();
  // local function
  auto get_step = [](double dv, double sv, int svc, double vmin, double vres,
                     int &stepV, double &tDeltaV, double &tMaxV) {
    if (dv > 0) {
      stepV = 1;
      tDeltaV = vres / dv;
      tMaxV = (vmin + (svc + 1) * vres - sv) / dv; // parametric distance until the first crossing
    } else if (dv < 0) {
      stepV = -1;
      tDeltaV = vres / -dv;
      tMaxV = (vmin + svc * vres - sv) / dv;
    } else {
      stepV = 0;
      tDeltaV = 0.0;
      tMaxV = std::numeric_limits<double>::infinity(); // the line doesn't cross the next plane
    }
  };

  // find start and end cells
  int sxc = meters2cells(sx, xmin, xres);
  int exc = meters2cells(ex, xmin, xres);
  int syc = meters2cells(sy, ymin, yres);
  int eyc = meters2cells(ey, ymin, yres);
  int szc = meters2cells(sz, zmin, zres);
  int ezc = meters2cells(ez, zmin, zres);
  double dx = ex - sx;
  double dy = ey - sy;
  double dz = ez - sz;
  int stepX, stepY, stepZ;          // direction of grid traversal
  double tDeltaX, tDeltaY, tDeltaZ; // parametric step size along different dimensions
  double tMaxX, tMaxY, tMaxZ; // used to determine the dim of the next step along the line
  get_step(dx, sx, sxc, xmin, xres, stepX, tDeltaX, tMaxX);
  get_step(dy, sy, syc, ymin, yres, stepY, tDeltaY, tMaxY);
  get_step(dz, sz, szc, zmin, zres, stepZ, tDeltaZ, tMaxZ);

  // Add initial voxel to the list
  xvec.push_back(sxc);
  yvec.push_back(syc);
  zvec.push_back(szc);
  while ((sxc != exc) || (syc != eyc) || (szc != ezc)) {
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        sxc += stepX;
        tMaxX += tDeltaX;
      } else {
        szc += stepZ;
        tMaxZ += tDeltaZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        syc += stepY;
        tMaxY += tDeltaY;
      } else {
        szc += stepZ;
        tMaxZ += tDeltaZ;
      }
    }
    xvec.push_back(sxc);
    yvec.push_back(syc);
    zvec.push_back(szc);
  }
}

inline Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>
inflate_map2d(const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &cmap,
              const std::vector<double> &res, double rad) {
  auto omap = cmap;
  if (rad < 0.00001)
    return omap; // nothing to do here

  int bbx = static_cast<int>(std::ceil(rad / res[0]));
  int bby = static_cast<int>(std::ceil(rad / res[1]));

  std::vector<std::array<int, 2>> ns;
  ns.reserve((2 * bbx + 1) * (2 * bby + 1));
  for (int nx = -bbx; nx <= bbx; ++nx)
    for (int ny = -bby; ny <= bby; ++ny) {
      double d = std::hypot(nx * res[0], ny * res[1]);
      if (0 < d && d <= rad)
        ns.push_back({nx, ny});
    }

  for (unsigned int x = 0; x < cmap.rows(); ++x)
    for (unsigned int y = 0; y < cmap.cols(); ++y) {
      if (cmap(x, y)) {
        for (const auto &it : ns) {
          int nx = x + it[0], ny = y + it[1];
          if (0 <= nx && nx < cmap.rows() && 0 <= ny && ny < cmap.cols())
            omap(nx, ny) = true;
        }
      }
    }
  return omap;
}


/*
 * TODO: not finished yet
inline std::vector<int> inflate_map( const std::vector<int>& cmap,
                                     const std::vector<int>& dim,
                                     const std::vector<double>& res,
                                     const std::vector<double>& rad )
{
  // find the number of cells to inflate per dimension
  std::vector<int> bb(res.size());
  int num_ns = 1;
  for( int i = 0; i < res.size(); ++i )
  {
    bb[i] = static_cast<int>(std::ceil(rad[i]/res[i]));
    num_ns *= 2*bb[i]+1;
  }

  // compute the neighbor coordinates
  std::vector<std::vector<int>> ns(num_ns);
  for( int i = 0; i < bb.size(); ++i )
    for( int neib = -bb[i]; neib <= bb[i]; ++neib)
      ns

  auto omap = cmap;
  for( int i = 0; i < cmap.size(); ++i )
    if( cmap[i] > 0 )
    {
      // find the neighbors and make them occupied

    }
  for( int i = 0; i < dim.size(); ++i )
    for( int x = 0; x < dim[i]; ++x )


}
*/
}
#endif
