//
// Created by brent on 9/3/18.
//

#ifndef INFO_GATHERING_LOGGER_H
#define INFO_GATHERING_LOGGER_H

#include <Eigen/Dense>
#include <list>
#include <map>

namespace nx {


    class Logger {

        using namespace Eigen;

    public:

        // Output buffers
        int max_targets = 0;
        std::vector<std::list<VectorXd>> y_est_list;
        std::vector<std::list<MatrixXd>> Sigma_est_list;
        std::vector<std::list<std::map<int, int>>> da_i;
        std::vector<std::list<std::map<int, int>>> da_ir;
        MatrixXd Ground_truth;
        MatrixXd Ground_truth_da;
        MatrixXd robot_pos;
        int n_robots;
        int Tmax;


        Logger(int n_robots, int n_targets, int target_dim, int Tmax) :
                y_est_list(std::vector<std::list<VectorXd>>(n_robots)),
                Sigma_est_list(std::vector<std::list<MatrixXd>>(n_robots)),
                da_i(std::vector<std::list<std::map<int, int>>>(n_robots)),
                da_ir(std::vector<std::list<std::map<int, int>>>(n_robots)),
                Ground_truth(MatrixXd(n_targets * target_dim, Tmax)),
                Ground_truth_da(MatrixXd(n_targets, Tmax)),
                robot_pos(MatrixXd(n_robots * 3, Tmax)),
                n_robots(n_robots),
                Tmax(Tmax) {


        }


        void LogData() {
            // Record data
//        for (int j = 0; j < robots.size(); j++) {
//            // Push back the current y_estimate, and Sigma estimates
//            y_est_list.at(j).push_back(robots[j].get_y_est());
//            //y_est_list.at(j).push_back(ENV.get_target());
//            Sigma_est_list.at(j).push_back(robots[j].get_Sigma_est());
//            da_i.at(j).push_back(robots[j].da);
//            da_ir.at(j).push_back(robots[j].da_reverse);
//            robot_pos.block(3 * j, t, 3, 1) = robots[j].x;
//            if (robots[j].get_da().size() > max_targets) {
//                max_targets = robots[j].get_da().size();
//            }
//        }
//        Ground_truth.col(t) = ENV.get_target();
//        // Fill in Environment DA
//        for (int j = 0; j < ENV.da_reverse.size(); j++) {
//            Ground_truth_da(j, t) = ENV.da_reverse[j];
//        }
//


        }


        void writeData() {
//
//            for (int i = 0; i < n_robots; i++) {
//                std::cout << "Final cov list: " << std::endl;
//                for (int j = 0; j < robots[i].da_reverse.size(); j++) {
//                    std::cout << robots[i].da_reverse[j] << " "
//                              << robots[i].get_Sigma_est().block(p.target_dim * j, p.target_dim * j, p.target_dim,
//                                                                 p.target_dim)
//                              << std::endl;
//                }
//            }
//            for (int i = 0; i < n_robots && Tmax > 20; i++) {
//                if (robots[i].get_y_est().rows() == ENV.get_target().rows())
//                    std::cout << "RMS Error of Estimate_" << i << ": "
//                              << (ENV.get_target() - robots[i].get_y_est()).norm()
//                              << std::endl;
//
//            }
//            std::cout << "Map visibility: " << 0.0 << std::endl;// robots[0].getPctSeen() << std::endl;
//
//            int num_dim_y_est = max_targets * p.target_dim;
//            for (int i = 0; i < n_robots; i++) {
//                int l = robots[i].get_y_est().rows();
//                if (l > num_dim_y_est)// This is because the estimate may have changed size
//                    num_dim_y_est = l;
//            }
//
//            // Now write the outputs to CSV files
//            int num_dim_y_est_2 = num_dim_y_est * num_dim_y_est;
//            MatrixXd y_est_return(num_dim_y_est * n_robots, Tmax);
//            MatrixXd Sigma_est_return(num_dim_y_est_2 * n_robots, Tmax);
//            MatrixXd da_return(num_dim_y_est / p.target_dim * n_robots, Tmax);
//
//            // Temp vectors
//            VectorXd y_t(num_dim_y_est);
//            VectorXd Sigma_t(num_dim_y_est_2);
//            VectorXd da_est(num_dim_y_est / p.target_dim);
//
//            // Write all the data to be returned here.
//            for (int i = 0; i < n_robots; i++) {
//                // Iterators
//                auto it = y_est_list.at(i).begin();
//                auto qt = da_i.at(i).begin();
//                auto mt = da_ir.at(i).begin();
//                int j = 0; // Timestep
//                // Loop over timestep
//                for (auto jt = Sigma_est_list.at(i).begin();
//                     it != y_est_list.at(i).end();
//                     ++jt, ++it, ++j, ++qt, ++mt) {
//                    VectorXd y_est = *it;
//
//                    MatrixXd Sigma_est_m = *jt;
//                    VectorXd Sigma_est(Map<VectorXd>(Sigma_est_m.data(), Sigma_est_m.rows() * Sigma_est_m.cols()));
//                    y_t.fill(-100.0);
//                    Sigma_t.fill(-100.0);
//
//                    y_t.segment(0, y_est.rows()) = y_est;
//                    Sigma_t.segment(0, Sigma_est.rows()) = Sigma_est;
//
//                    // Write y_est and Sigma _est
//                    y_est_return.block(i * num_dim_y_est, j, num_dim_y_est, 1) = y_t;
//                    Sigma_est_return.block(i * num_dim_y_est_2, j, num_dim_y_est_2, 1) = Sigma_t;
//
//                    if (p.debug && 0) {
//                        std::cout << "At time " << j << " robot " << i << " observed: " << *it << std::endl;
//                        std::cout << "At time " << j << " robot " << i << " observed: " << y_t << std::endl;
//                        std::cout << "At time " << j << " robot " << i << " observed: " <<
//                                  y_est_return.block(i * num_dim_y_est, j, num_dim_y_est, 1) << std::endl;
//                    }
//                    // Write the DA
//                    std::map<int, int> da_tr = *mt;
//                    da_est.fill(-100.0);
//
//                    for (int k = 0; k < da_tr.size(); k++) {
//                        da_est(k, 0) = da_tr[k];
//                    }
//                    // Write da
//                    da_return.block(i * num_dim_y_est / p.target_dim, j, num_dim_y_est / p.target_dim, 1) = da_est;
//                }
//            }
//
//            if (p.log) {
//                // Write to CSV
//                std::string root_path = p.output_file;
//                std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
//                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
//
//                std::ostringstream oss;
//                oss << std::put_time(std::localtime(&now_c), "%F %T");
//                std::string y_est_file = root_path + "/y_est.csv";
//                std::string Sigma_est_file = root_path + "/Sigma_est.csv";
//                std::string da_est_file = root_path + "/da_est.csv";
//                std::string ground_truth_file = root_path + "/ground_truth.csv";
//                std::string ground_truth_da_file = root_path + "/ground_truth_da.csv";
//                std::string vis_file = root_path + "/vis.csv";
//                std::string robot_pos_file = root_path + "/pos.csv";
//
//
//                std::cout << "Writing to file root: " << y_est_file << "\n";
//                nx::writeToCSVfile(y_est_file, y_est_return);
//                nx::writeToCSVfile(Sigma_est_file, Sigma_est_return);
//                nx::writeToCSVfile(da_est_file, da_return);
//                nx::writeToCSVfile(vis_file, Vis);
//                nx::writeToCSVfile(robot_pos_file, robot_pos);
//                nx::writeToCSVfile(ground_truth_file, Ground_truth);
//                nx::writeToCSVfile(ground_truth_da_file, Ground_truth_da);
//
//
//            }
//
//
//        }
        }
    }; // end class
} // end namespace
#endif //INFO_GATHERING_LOGGER_H
