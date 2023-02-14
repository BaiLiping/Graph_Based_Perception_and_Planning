#ifndef __UTILS_NX_H_
#define __UTILS_NX_H_

#include <Eigen/Core>
#include <fstream>
#include <cmath>
#include <limits>
#include <chrono>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <random>
//#include <iostream>
//#include <iterator>     // std::ostream_iterator
//#include <algorithm>    // std::copy
#ifndef PI
#define PI 3.1415962653
#endif
namespace nx {





    /*
     * CSV Format
     *
     */
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
    /*
     * Timing
     */
    static std::chrono::high_resolution_clock::time_point __last_tic__;

    static std::chrono::high_resolution_clock::time_point &tic() {
        __last_tic__ = std::chrono::high_resolution_clock::now();
        return __last_tic__;
    }

    static double toc() {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - __last_tic__).count();
        //return std::chrono::duration_cast<std::chrono::seconds>( std::chrono::high_resolution_clock::now() - __last_toc__ ).count();
    }

    static double toc(std::chrono::high_resolution_clock::time_point &t2) {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t2).count();
    }

    /*
     * Probability
     */
    // call the following once:
    // srand((unsigned)time(NULL));
    // before calling a loop of this:
    // Returns a random integer in [a, b]
    inline int uniform_int(int a, int b) {
        return rand() % (b - a + 1) + a;
    }

    // call the following once:
    // srand((unsigned)time(NULL));
    // before calling a loop of this:
    // Returns a random double in [a, b]
    inline double uniform_cont(double a, double b) {
        return a + (b - a) * ((double) rand() / (double) RAND_MAX);
    }

    /*
     * Matrices
     */
    template<typename T>
    inline int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    // returns the first odd integer larger than x


    template<typename Derived>
    inline bool is_finite(const Eigen::MatrixBase<Derived> &x) {
        return ((x - x).array() == (x - x).array()).all();
    }

    template<typename Derived>
    inline bool is_nan(const Eigen::MatrixBase<Derived> &x) {
        return ((x.array() == x.array())).all();
    }

    inline Eigen::MatrixXd yamlInitMatrix(std::vector<std::vector<double>> Matrix) {
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(Matrix.size(), Matrix.at(0).size());
        for (int i = 0; i < Matrix.size(); i++) {
            for (int j = 0; j < Matrix.at(0).size(); j++) {
                mat(i, j)= Matrix.at(i).at(j);
            }
        }
        return mat;
    }
    /*
     * Rotations
     */
    inline double deg2rad(double deg) { return (M_PI / 180.0) * deg; }

    inline double rad2deg(double rad) { return (180.0 / M_PI) * rad; }

    // [x,y,z] = sph2cart(az,elev,r)
    template<typename T>
    inline void sph2cart(Eigen::Matrix<T, 3, 1> const &az_elev_rad, Eigen::Matrix<T, 3, 1> &x_y_z) {
        x_y_z.x() = az_elev_rad.z() * std::cos(az_elev_rad.y()) * std::cos(az_elev_rad.x());
        x_y_z.y() = az_elev_rad.z() * std::cos(az_elev_rad.y()) * std::sin(az_elev_rad.x());
        x_y_z.z() = az_elev_rad.z() * std::sin(az_elev_rad.y());
    }

    template<typename T>
    inline void cart2sph(Eigen::Matrix<T, 3, 1> const &x_y_z, Eigen::Matrix<T, 3, 1> &az_elev_rad) {
        az_elev_rad.x() = std::atan2(x_y_z.y(), x_y_z.x());
        az_elev_rad.y() = std::atan2(x_y_z.z(), std::hypot(x_y_z.x(), x_y_z.y()));
        az_elev_rad.z() = std::hypot(std::hypot(x_y_z.x(), x_y_z.y()), x_y_z.z());
    }

    inline double restrict_angle(double phi, double minrange = -M_PI, double maxrange = M_PI) {
        // return minrange + std::fmod( (phi - minrange), (maxrange - minrange));
        // NOTE!!: fmod does not behave like MATLAB mod!
        double x = phi - minrange;
        double y = maxrange - minrange;
        return minrange + x - y * std::floor(x / y);
    }

    inline void smart_plus_SE2(double x1, double y1, double q1,
                               double x2, double y2, double q2,
                               double &x, double &y, double &q) {
        double sq1 = std::sin(q1);
        double cq1 = std::cos(q1);
        x = x1 + cq1 * x2 - sq1 * y2;
        y = y1 + sq1 * x2 + cq1 * y2;
        q = nx::restrict_angle(q1 + q2);
    }

    inline Eigen::Matrix3d rotx(double roll) {
        Eigen::Matrix3d Rx;
        Rx << 1, 0, 0,
                0, cos(roll), -sin(roll),
                0, sin(roll), cos(roll);
        return Rx;
    }

    inline Eigen::Matrix3d roty(double pitch) {
        Eigen::Matrix3d Ry;

        Ry << cos(pitch), 0, sin(pitch),
                0, 1, 0,
                -sin(pitch), 0, cos(pitch);
        return Ry;
    }

    inline Eigen::Matrix3d rotz(double yaw) {
        Eigen::Matrix3d Rz;
        Rz << cos(yaw), -sin(yaw), 0,
                sin(yaw), cos(yaw), 0,
                0, 0, 1;
        return Rz;
    }

    inline double wrap(double angle1, double angle2) {
        double result = angle1 - angle2;
        if (result > PI) {
            result = 2*PI  - result;
        }
        else if (result < -PI) {
            result = 2*PI + result;
        }
        return result;

    }
    // XXX: ASSUMES THE SCALAR IS THE LAST COMPONENT!
    inline void quat2rot(const Eigen::Vector4d &q, Eigen::Matrix<double, 3, 3> &R) {
        R << 1.0 - 2.0 * (q(1) * q(1) + q(2) * q(2)),
                2.0 * q(0) * q(1) - 2.0 * q(3) * q(2),
                2.0 * q(3) * q(1) + 2.0 * q(0) * q(2),
                2.0 * q(0) * q(1) + 2.0 * q(3) * q(2),
                1.0 - 2.0 * (q(0) * q(0) + q(2) * q(2)),
                2.0 * q(1) * q(2) - 2.0 * q(3) * q(0),
                2.0 * q(0) * q(2) - 2.0 * q(3) * q(1),
                2.0 * q(1) * q(2) + 2.0 * q(3) * q(0),
                1.0 - 2.0 * (q(0) * q(0) + q(1) * q(1));
    }


    inline double SO3_metric(const Eigen::Vector4d &q1, const Eigen::Vector4d &q2) {
        double quat_inner_prod = q1.x() * q2.x() + q1.y() * q2.y() + q1.z() * q2.z() + q1.w() * q2.w();
        return std::acos(2 * quat_inner_prod * quat_inner_prod - 1);
    }

    template<typename T>
    inline T quatnorm(Eigen::Matrix<T, 4, 1> const &quat) {
        return (std::pow(quat.w(), 2) + std::pow(quat.x(), 2) + std::pow(quat.y(), 2) + std::pow(quat.z(), 2));
    }

    template<typename T>
    inline T quatmod(Eigen::Matrix<T, 4, 1> const &quat) { return std::sqrt(quatnorm(quat)); }

    template<typename T>
    inline void quatnormalize(Eigen::Matrix<T, 4, 1> const &q_in, Eigen::Matrix<T, 4, 1> &q_out) {
        T mod = quatmod(q_in);
        q_out.x() = q_in.x() / mod;
        q_out.y() = q_in.y() / mod;
        q_out.z() = q_in.z() / mod;
        q_out.w() = q_in.w() / mod;
    }

    // Assumes ZYX rotation;
    template<typename T>
    inline void angle2quat(Eigen::Matrix<T, 3, 1> const &yaw_pitch_roll, Eigen::Matrix<T, 4, 1> &quat) {
        T cyaw = cos(yaw_pitch_roll.x() / 2);
        T cpitch = cos(yaw_pitch_roll.y() / 2);
        T croll = cos(yaw_pitch_roll.z() / 2);

        T syaw = sin(yaw_pitch_roll.x() / 2);
        T spitch = sin(yaw_pitch_roll.y() / 2);
        T sroll = sin(yaw_pitch_roll.z() / 2);

        quat.x() = cyaw * cpitch * sroll - syaw * spitch * croll;
        quat.y() = cyaw * spitch * croll + syaw * cpitch * sroll;
        quat.z() = syaw * cpitch * croll - cyaw * spitch * sroll;
        quat.w() = cyaw * cpitch * croll + syaw * spitch * sroll;
    }

    // Assumes ZYX rotation
    template<typename T>
    inline void quat2angle(Eigen::Matrix<T, 4, 1> const &quat, Eigen::Matrix<T, 3, 1> &yaw_pitch_roll) {
        Eigen::Matrix<T, 4, 1> quat_n;
        quatnormalize(quat, quat_n);

        T r11 = 2 * (quat_n.x() * quat_n.y() + quat_n.w() * quat_n.z());
        T r12 = std::pow(quat_n.w(), 2) + std::pow(quat_n.x(), 2) - std::pow(quat_n.y(), 2) - std::pow(quat_n.z(), 2);
        T r21 = -2 * (quat_n.x() * quat_n.z() - quat_n.w() * quat_n.y());
        T r31 = 2 * (quat_n.y() * quat_n.z() + quat_n.w() * quat_n.x());
        T r32 = std::pow(quat_n.w(), 2) + std::pow(quat_n.z(), 2) - std::pow(quat_n.x(), 2) - std::pow(quat_n.y(), 2);

        // truncate r21 if above the allowed range
        if (abs(r21) >= static_cast<T>(1.0))
            r21 = nx::sgn(r21) * static_cast<T>(1.0);

        yaw_pitch_roll.x() = std::atan2(r11, r12);
        yaw_pitch_roll.y() = std::asin(r21);
        yaw_pitch_roll.z() = std::atan2(r31, r32);
    }

    /*
     * File Management
     */
    inline std::ofstream &open_out_file(std::ofstream &fsr, const std::string &file) {
        // put file in valid state first
        fsr.close();    // close in case it was already open
        fsr.clear();    // clear any existing errors
        fsr.open(file.c_str());
        return fsr;
    }

    inline std::ifstream &open_in_file(std::ifstream &fsr, const std::string &file) {
        // put file in valid state first
        fsr.close();    // close in case it was already open
        fsr.clear();    // clear any existing errors
        fsr.open(file.c_str());
        return fsr;
    }

    inline Eigen::MatrixXd readFromCSVfile(std::string name, int rows, int cols) {

        std::ifstream in(name);

        std::string line;

        int row = 0;
        int col = 0;

        Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

        if (in.is_open()) {

            while (std::getline(in, line)) {

                char *ptr = (char *) line.c_str();
                int len = line.length();

                col = 0;

                char *start = ptr;
                for (int i = 0; i < len; i++) {

                    if (ptr[i] == ',') {
                        res(row, col++) = atof(start);
                        start = ptr + i + 1;
                    }
                }
                res(row, col) = atof(start);

                row++;
            }

            in.close();
        }
        return res;
    }


    inline void writeToCSVfile(std::string name, Eigen::MatrixXd matrix)
    {


//        boost::filesystem::path dir(name);
//
//        if(!(boost::filesystem::exists(dir.parent_path()))){
//            std::cout<<"Directory Doesn't Exist"<<std::endl;
//
//            if (boost::filesystem::create_directory(dir.parent_path()))
//                std::cout << "....Successfully Created !" << std::endl;
//        }
        std::ofstream file(name.c_str());
        file << matrix.format(CSVFormat);
        file.close();

    }



    // TODO Fix the design of this..
    class Probability
    {

    public:
        std::random_device rd;
        std::mt19937 gen;

        Probability() {
            gen = std::mt19937(rd());
        }
    };


    struct normal_random_variable
    {
        normal_random_variable(Eigen::MatrixXd const& covar)
                : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
        {}

        normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
                : mean(mean)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
            transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
        }

        Eigen::VectorXd mean;
        Eigen::MatrixXd transform;

        Eigen::VectorXd operator()() const
        {
            static std::mt19937 gen{ std::random_device{}() };
            static std::normal_distribution<> dist;

            return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](double x) { return dist(gen); });
        }
    };


    static Probability p;

    inline Eigen::VectorXd normal_dist(Eigen::VectorXd mean, Eigen::MatrixXd cov)
    {
        normal_random_variable sample {mean, cov};
        return sample().cast<double>();
    }

    inline Eigen::VectorXd normal_dist(Eigen::MatrixXd cov)
    {
        normal_random_variable sample {cov};
        return sample().cast<double>();
    }

    // Scalar normal dist
    inline double normal_dist(double mean, double cov)
    {
        std::normal_distribution<double> dist = std::normal_distribution<double>(mean, cov);
        return dist(p.gen);
    }
}

#endif