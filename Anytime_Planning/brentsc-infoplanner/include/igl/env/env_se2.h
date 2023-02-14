//
// Created by brent on 4/24/17.
//

#ifndef NX_ASTAR_ENV_TARGET_SE2_H
#define NX_ASTAR_ENV_TARGET_SE2_H

#include <memory>   // std::unique_ptr
#include <utility>      // std::pair
#include <igl/env/env_int.h>
#include <igl/mapping/map_nx.h>
#include <igl/control/motion_primitive.h>
#include <igl/control/dd_motion_model.h>
#include <iostream>
#include <igl/se3_pose.h>

namespace nx {

class SE2Environment : public Environment<SE3Pose> {
 protected:
  SE3Pose goal_coord;     // discrete coordinates of the goal node
  typedef MotionPrimitive<std::array<double, 3>, std::array<double, 2>> MPrim;
  size_t max_len_traj_ = 0;
  double samp = 0.5; // TODO Correct this for usage with motion primitives
  int yaw_discretization_size = 60; // Discretize the interval [-pi, pi].
  double yaw_res = 2 * PI / yaw_discretization_size;

  bool is_3d; // Check if collision checking should be done in 3D.

  // for each orientation
  //   for each motion primitive
  //     for each segment
  //      we have a vector of micro states (discrete coordinates)
  std::vector<std::vector<std::vector<std::vector<std::array<int, 3>>>>> mprim_xd_;

 public:

  SE2Environment(const nx::map_nd &map_in,
                 const std::vector<char> &cmap_ptr,
                 const std::string &mprim_yaml,
                 SE3Pose goal = SE3Pose())

      : Environment<SE3Pose>(map_in, cmap_ptr, goal) {

    // Initialize motion primitives
    nx::mprmsFromYAML<std::array<double, 3>, std::array<double, 2>>(mprim_yaml,
                                                                    static_cast<std::array<double, 3> (*)(
                                                                        const std::array<double, 3> &,
                                                                        const std::array<double, 2> &,
                                                                        double)>(nx::dd_motion_model),
                                                                    std::array<double, 3>({0, 0, 0}), mprim_,
                                                                    samp);

    // Check if this is a 3D Map or not.
    is_3d = map_in.res().size() == 3;


//                std::cout <<"Motion Primitives " << std::endl;
//                for( int mp = 0; mp < mprim_.size(); ++mp)
//                {
//                  for( int l = 0; l < mprim_[mp].xVecVec.size(); ++l )
//                    for( int m = 0; m < mprim_[mp].xVecVec[l].size(); ++m )
//                      std::cout << mprim_[mp].xVecVec[l][m][0] << " "
//                                << mprim_[mp].xVecVec[l][m][1] << " "
//                                << mprim_[mp].xVecVec[l][m][2] << std::endl;
//                  std::cout << std::endl;
//                  std::cout << std::endl;
//                }

    // compute discrete coordinates for each orientation
    init_mprim_xd_();

    for (int tr = 0; tr < mprim_.size(); ++tr)
      if (mprim_[tr].uVec.size() > max_len_traj_)
        max_len_traj_ = mprim_[tr].uVec.size();
  }

  int state_to_idx(const SE3Pose &state) const override {
    // First we convert the XYZ position to cell coordinates.
    std::vector<int> cell = state_to_cell(state);
    return map->subv2ind_colmajor(cell);
  }


  Vector3d state_to_SE2(const SE3Pose &state) const override {
    return state.getSE2();
  }

  std::vector<int> state_to_cell(const SE3Pose &state) const override {
    Vector3d position = state.position;
    return map->meters2cells({position[0], position[1]});
  }

  inline void get_succ(const SE3Pose &curr,
                       std::vector<SE3Pose> &succ,
                       std::vector<double> &succ_cost,
                       std::vector<int> &action_idx) const override {

    // Get constant values from SE3Pose
    double z = curr.position[2]; // Z coordinate.
    double yaw = curr.getYaw();
    double pitch = curr.getPitch();
    double roll = curr.getRoll();

    size_t num_traj = mprim_.size();
    //size_t len_traj = mprim_[0].xVecVec.size();
    //size_t len_traj_fine = mprim_[0].(*mp_x_ptr_)[0].size();
    //size_t num_samp_vals = len_traj_fine / len_traj;

    double min_x = map->min()[0];
    double res_x = map->res()[0];
    int sz_x = map->size()[0];
    double min_y = map->min()[1];
    double res_y = map->res()[1];
    int sz_y = map->size()[1];

    // Reserve space for successors
    succ.reserve(num_traj * max_len_traj_);
    succ_cost.reserve(num_traj * max_len_traj_);
    action_idx.reserve(num_traj * max_len_traj_);

    for (unsigned tr = 0; tr < num_traj; ++tr) {
      // Get current cell Theta
      double theta = curr.getYaw();
      int cell_theta = nx::meters2cells(theta, -PI, yaw_res);

//                std::cout <<"Double theta: " << theta << std::endl;
//                std::cout <<"Cell theta: " << cell_theta << std::endl;


      std::vector<int> cell = map->meters2cells({curr.position[0], curr.position[1], curr.position[2]});
      size_t len_traj = mprim_xd_[cell_theta][tr].size();

      for (unsigned len = 0; len < len_traj; ++len) {
        // check for collisions along the microstates
        bool collided = false;

        unsigned int len_fine_sz = mprim_xd_[cell_theta][tr][len].size();

        for (unsigned int len_fine = 0; len_fine < len_fine_sz; ++len_fine) {

          int x_val = cell[0] + mprim_xd_[cell_theta][tr][len][len_fine][0];
          int y_val = cell[1] + mprim_xd_[cell_theta][tr][len][len_fine][1];
          int z_val = cell[2];

          //int q_val = mprim_xd_[curr[2]][tr][len][len_fine][2];

          // Discard motion primitives that go outside of the map
          if (x_val < 0 || x_val >= sz_x ||
              y_val < 0 || y_val >= sz_y) {
            collided = true;
            break;
          }

          // Discard motion primitives that collide
          int lindix;
          if (is_3d) // If the map is 3D, we check 3D space for collisions.
            lindix = map->subv2ind_colmajor({x_val, y_val, z_val});
          else
            lindix = map->subv2ind_colmajor({x_val, y_val});

          if (cmap_->at(lindix) == '1') {
//            std::cout <<"Collided at index:" << lindix << " with cmap val " << cmap_->at(lindix) << std::endl;
            collided = true;
            break;
          }

          if (len_fine == len_fine_sz - 1) // The end of the fine trajectory
          {
            int q_val = mprim_xd_[cell_theta][tr][len][len_fine][2];
            std::vector<double> meters = map->cells2meters({x_val, y_val});

            double new_yaw = nx::cells2meters(q_val, -PI, yaw_res);
//                            std::cout <<"New qval " << q_val << std::endl;
//                            std::cout <<"New yaw: " << new_yaw << std::endl;
            Vector3d new_position({meters[0], meters[1], z});
            double yaw_change = restrict_angle(new_yaw - yaw);
            Matrix3d rotation = rotz(yaw_change);
            Matrix3d new_orientation = curr.orientation * rotation;

            succ.push_back(SE3Pose(new_position, new_orientation));
            succ_cost.push_back(mprim_[tr].cVec[len]);
            action_idx.push_back(tr * max_len_traj_ + len);  // action_id
          }
        }
        // No need to check the rest of the trajectory length if we collided already
        if (collided)
          break;
      }
    }
  }

  void forward_action(const SE3Pose &curr, int action_id,
                      std::vector<SE3Pose> &next_micro) const override {


    // Get current cell Theta
    double theta = curr.getYaw();
    int cell_theta = nx::meters2cells(theta, -PI, yaw_res);
    std::vector<int> cell = map->meters2cells({curr.position[0], curr.position[1]});
    //    std::cout <<" Coord"  << curr[0] << " " << curr[1] << " " << curr[2] <<  std::endl;
    size_t num_traj = mprim_.size();
    //size_t len_traj_fine = mprim_[0].(*mp_x_ptr_)[0].size();
    //size_t num_samp_vals = len_traj_fine / len_traj;


    // find which trajectory it is
    size_t tr = action_id / max_len_traj_;
    size_t len_sz = action_id % max_len_traj_;

    for (size_t len = 0; len <= len_sz; ++len) {
      size_t len_fine_sz = mprim_xd_[cell_theta][tr][len].size();
      for (size_t len_fine = 0; len_fine < len_fine_sz; ++len_fine) {
        int x_val = cell[0] + mprim_xd_[cell_theta][tr][len][len_fine][0];
        int y_val = cell[1] + mprim_xd_[cell_theta][tr][len][len_fine][1];
        int q_val = mprim_xd_[cell_theta][tr][len][len_fine][2];

          // Construct the new SE3Pose

          std::vector<double> meters = map->cells2meters({x_val, y_val});

          double new_yaw = nx::cells2meters(q_val, -PI, yaw_res);
          Vector3d new_position({meters[0], meters[1], curr.position[2]});
          double yaw_change = restrict_angle(new_yaw - curr.getYaw());
          Matrix3d rotation = rotz(yaw_change);
          Matrix3d new_orientation = rotation * curr.orientation;

          next_micro.push_back(SE3Pose(new_position, new_orientation));

        //int x_val = nx::meters2cells( nx::cells2meters( curr[0], MAP_ptr->min()[0], MAP_ptr->res()[0] )
        //  + (*mprim_[tr].mp_x_ptr)[len][len_fine].first, MAP_ptr->min()[0], MAP_ptr->res()[0]);
        //int y_val = nx::meters2cells( nx::cells2meters( curr[1], MAP_ptr->min()[1], MAP_ptr->res()[1] )
        //  + (*mprim_[tr].mp_x_ptr)[len][len_fine].second, MAP_ptr->min()[1], MAP_ptr->res()[1]);
      }
    }
  }

 private:
  void init_mprim_xd_() {
    double x, y, q;
    int xc, yc, qc;
    int theta_sz = (yaw_discretization_size);
    mprim_xd_.resize(theta_sz);
    int num_prim = mprim_.size();
    for (unsigned k = 0; k < theta_sz; ++k) {
      mprim_xd_[k].resize(num_prim);
      double ori = nx::cells2meters(k, -PI, yaw_res);
      for (unsigned pr = 0; pr < num_prim; ++pr) {
        int num_seg = mprim_[pr].xVecVec.size();
        mprim_xd_[k][pr].resize(num_seg);
        for (unsigned seg = 0; seg < num_seg; ++seg) {
          int num_sta = mprim_[pr].xVecVec[seg].size();
          for (unsigned st = 0; st < num_sta; ++st) {
            nx::smart_plus_SE2(map->origin()[0], map->origin()[1], ori,
                               mprim_[pr].xVecVec[seg][st][0],
                               mprim_[pr].xVecVec[seg][st][1],
                               mprim_[pr].xVecVec[seg][st][2],
                               x, y, q);
            xc = nx::meters2cells(x, map->min()[0], map->res()[0])
                - map->origincells()[0];
            yc = nx::meters2cells(y, map->min()[1], map->res()[1])
                - map->origincells()[1];
            qc = nx::meters2cells(q, -PI, yaw_res);
            // add only if unique
            if (mprim_xd_[k][pr][seg].empty() ||
                xc != mprim_xd_[k][pr][seg].back()[0] ||
                yc != mprim_xd_[k][pr][seg].back()[1] ||
                qc != mprim_xd_[k][pr][seg].back()[2]) {
              mprim_xd_[k][pr][seg].push_back({xc, yc, qc});
            }
          }
        }
      }
    }
  } // End function
 public:
  /**
   * Computes the distance metric between two states.
   * @return The distance between the two states.
   */
  double ComputeStateMetric(const SE3Pose &s1 , const SE3Pose & s2) const override
  {
    // First project into SE2
    Vector3d s1_se2 = s1.getSE2();
    Vector3d s2_se2 = s2.getSE2();

    Vector3d state_diff = s1_se2 - s2_se2;
    state_diff(2, 0) = nx::restrict_angle(state_diff(2, 0)); // Wrap Angle
    Vector4d state_diff_c; // Complex numbers for angle
    state_diff_c << state_diff(0, 0),
        state_diff(1, 0),
        std::cos(state_diff(2, 0)),
        std::sin(state_diff(2, 0));
    return state_diff_c.norm();

  };


}; // End class

} // End namespace


#endif //NX_ASTAR_ENV_TARGET_SE2_H
