#ifndef INFO_GATHERING_CAMERAMODEL_H
#define INFO_GATHERING_CAMERAMODEL_H

#include <Eigen/Dense>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <random>
#include <map>
#include <fstream>   // std::ifstream

#include <igl/sensing/sensor.h>

namespace nx
{
  // Camera-to-Optical frame transformation
  const Eigen::Matrix3d oRc_((Eigen::Matrix3d() << 0.0, -1.0,  0.0,
                                                   0.0,  0.0, -1.0,
                                                   1.0,  0.0,  0.0).finished());

  /*
   * @brief: Returns the pixel observation of 3-D point y from camera pose (p,R)
   *
   * @Input:
   *    R = 3 x 3 = camera orientation in the world FLU frame (x forward, y left, z up)
   *    p = 3 x 1 = camera position in the world FLU frame
   *    y = 3 x 1 = observed point position in the world FLU frame
   *    K = 2 x 3 = intrinsic calibration
   *
   *   K = [ f*su, f*st, cu]
   *       [    0, f*sv, cv]
   *
   *   where
   *     f = focal scaling in meters
   *     su = pixels per meter
   *     sv = pixels per meter
   *     cu = image center
   *     cv = image center
   *     st = nonrectangular pixel scaling
   *
   * @Output:
   *   z = 2 x 1 = pixel coordiantes of the point y
   */  
  inline Eigen::Vector2d cameraModel( const Eigen::Matrix3d& R, const Eigen::Vector3d& p,
                                      const Eigen::Vector3d& y, const Eigen::Matrix<double, 2,3>& K)
  {
    Eigen::Vector3d y_o = oRc_*R.transpose()*(y-p); // point in the camera optical frame
    return K*y_o/y_o.z();
  }
  
  inline Eigen::Matrix<double, 2, 3> projectionJacobian( const Eigen::Vector3d& v )
  {
    Eigen::Matrix<double, 2, 3> J;
    J << 1.0/v.z(), 0.0, -v.x()/v.z()/v.z(),
         0.0, 1.0/v.z(), -v.y()/v.z()/v.z();
    return J;
  }
  
  inline Eigen::Matrix<double, 2, 3> cameraModelJacobianY( const Eigen::Matrix3d& R, const Eigen::Vector3d& p,
                                                           const Eigen::Vector3d& y, const Eigen::Matrix<double, 2,3>& K)
  {
    Eigen::Matrix3d oRw = oRc_*R.transpose();
    return K.block<2,2>(0,0) * projectionJacobian(oRw*(y-p)) * oRw;
  }              
  inline Eigen::Matrix<double, 2, 3> cameraModelJacobianP( const Eigen::Matrix3d& R, const Eigen::Vector3d& p,
                                                           const Eigen::Vector3d& y, const Eigen::Matrix<double, 2,3>& K)
  {
    Eigen::Matrix3d oRw = oRc_*R.transpose();
    return - K.block<2,2>(0,0) * projectionJacobian(oRw*(y-p)) * oRw;
  }

	inline Eigen::Matrix3d skewSymmetricMatrix(const Eigen::Vector3d& x)
	{
    Eigen::Matrix3d S;
    S <<   0,  -x(2), x(1),
          x(2),  0,  -x(0),
         -x(1), x(0),  0;
    return S;
  }
  
	inline Eigen::Matrix<double,9,3> skewSymmetricMatrixJacobian( )
	{
	  Eigen::Matrix<double,9,3> J;
	  J <<  0.0,  0.0,  0.0,
	        0.0,  0.0,  1.0,
	        0.0, -1.0,  0.0,
	        0.0,  0.0, -1.0,
	        0.0,  0.0,  0.0,
	        1.0,  0.0,  0.0,
	        0.0,  1.0,  0.0,
	       -1.0,  0.0,  0.0,
	        0.0,  0.0,  0.0;
	  return J;
	}
	  
  inline Eigen::Matrix3d SO3RightJacobian( const Eigen::Vector3d& theta )
  {
    double theta_norm = theta.norm();
    double theta_norm2 = theta_norm*theta_norm;
    double theta_norm3 = theta_norm*theta_norm2;
    Eigen::Matrix3d theta_hat = skewSymmetricMatrix(theta);
    return Eigen::Matrix3d::Identity() - (1.0-std::cos(theta_norm))/theta_norm2*theta_hat + (theta_norm - std::sin(theta_norm))/theta_norm3*theta_hat*theta_hat;
  }

  inline Eigen::Matrix3d SO3RightJacobianInverse( const Eigen::Vector3d& theta )
  {
    double theta_norm = theta.norm();
    double theta_norm2 = theta_norm*theta_norm;
    Eigen::Matrix3d theta_hat = skewSymmetricMatrix(theta);
    return Eigen::Matrix3d::Identity() + theta_hat/2.0 + (1.0/theta_norm2 - (1.0 + std::cos(theta_norm))/2.0/theta_norm/std::sin(theta_norm))*theta_hat*theta_hat;
  }

  inline Eigen::Matrix3d SO3LeftJacobian( const Eigen::Vector3d& theta )
  {
    double theta_norm = theta.norm();
    double theta_norm2 = theta_norm*theta_norm;
    double theta_norm3 = theta_norm*theta_norm2;
    Eigen::Matrix3d theta_hat = skewSymmetricMatrix(theta);
    return Eigen::Matrix3d::Identity() + (1.0-std::cos(theta_norm))/theta_norm2*theta_hat + (theta_norm - std::sin(theta_norm))/theta_norm3*theta_hat*theta_hat;
  }

  inline Eigen::Matrix3d SO3LeftJacobianInverse( const Eigen::Vector3d& theta )
  {
    double theta_norm = theta.norm();
    double theta_norm2 = theta_norm*theta_norm;
    Eigen::Matrix3d theta_hat = skewSymmetricMatrix(theta);
    return Eigen::Matrix3d::Identity() - theta_hat/2.0 + (1.0/theta_norm2 - (1.0 + std::cos(theta_norm))/2.0/theta_norm/std::sin(theta_norm))*theta_hat*theta_hat;
  }

  inline Eigen::Matrix<double, 2, 3> cameraModelJacobianTheta( const Eigen::Matrix3d& R, const Eigen::Vector3d& p,
                                                               const Eigen::Vector3d& y, const Eigen::Matrix<double, 2,3>& K)
  {
    Eigen::Matrix3d oRw = oRc_*R.transpose();
    Eigen::AngleAxisd ThetaAA(R);
    Eigen::Vector3d Theta = ThetaAA.angle()*ThetaAA.axis();
    return K.block<2,2>(0,0) * projectionJacobian(oRw*(y-p)) * oRw * skewSymmetricMatrix(y-p) * SO3RightJacobian(-Theta);
  }    
          

  
  class CameraSensor : public Sensor<SE3Pose>
  {
    /*
     *
     *   K = [ f*su, f*st, cu]
     *       [    0, f*sv, cv]
     *
     *   where
     *     f = focal scaling in meters
     *     su = pixels per meter
     *     sv = pixels per meter
     *     cu = image center
     *     cv = image center
     *     st = nonrectangular pixel scaling
     */
    int height_, width_;           // camera resolution in pixels
    Eigen::Matrix<double, 2,3> K_; // camera calibration matrix   
    const double noise_stdev_;
    
    // Noise generation
    std::random_device rd_;
    std::default_random_engine gen_;
    std::normal_distribution<double> dis_;
    
    typedef CGAL::Simple_cartesian<double>::Point_3 Point;
    typedef CGAL::Simple_cartesian<double>::Segment_3 Segment;
    typedef CGAL::Surface_mesh<Point> Mesh;
    typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
    typedef CGAL::AABB_traits<CGAL::Simple_cartesian<double>, Primitive> Traits;
    typedef CGAL::AABB_tree<Traits> Tree;
    Mesh mesh_;
    Tree tree_;
    bool mapSet_;
    
  public:
    CameraSensor( int height, int width,
                  const Eigen::Matrix<double, 2,3>& K,
                  double noise_stdev, std::string off_file = "")
      : Sensor(2), height_(height), width_(width), K_(K),
        noise_stdev_(noise_stdev), gen_(rd_()), dis_(0.0,noise_stdev_),
        mapSet_(off_file.length() > 0)
    {
        if (mapSet_)
            setMap(off_file);
    }
    
    void setMap( const std::string& off_file_name )
    {
      std::ifstream input(off_file_name);
      mesh_.clear();
      input >> mesh_;
      if( mesh_.number_of_faces() > 0 )
      {
        tree_.rebuild(faces(mesh_).first, faces(mesh_).second, mesh_);
        tree_.build();
        tree_.accelerate_distance_queries();
        mapSet_ = true;
      }
    }

    bool isValid( const Eigen::Vector2d& z ) const
    {
      return (0.0 < z(0) && z(0) < static_cast<double>(width_) && 
              0.0 < z(1) && z(1) < static_cast<double>(height_));
    }

    Eigen::MatrixXd sense( const Eigen::Vector4d& x,
                           const TargetModel &tmm ) const
    {
      Eigen::Vector3d p(x(0),x(1),x(3));
      Eigen::Matrix3d Rz;
      Rz << std::cos(x(2)), -std::sin(x(2)), 0,
            std::sin(x(2)), std::cos(x(2)), 0,
                0, 0, 1;    

      Eigen::VectorXd y_true = tmm.GetTargetState();
      int y_dim = tmm.dim;
      int num_targets = static_cast<int>(y_true.rows()) / y_dim;
      
      // Measurements generated are a 4-tuple (pixelx, pixely, data association, detectable).
      Eigen::MatrixXd z(num_targets, 4);
      std::map<int, int> da = tmm.mgr.da_reverse; // This accepts lookup by target index rather than ID
      for (int i = 0; i < num_targets; i++)
      {
        Eigen::Vector3d y = y_true.segment(y_dim * i, 3);
        Eigen::Vector2d pix = nx::cameraModel( Rz, p, y, K_);
        z(i,2) = da[i];
        z(i,3) = static_cast<double>(isValid(pix));
        if( mapSet_ && z(i,3) > 0.0 )
        {
          Segment segment_query(Point(p.x(),p.y(),p.z()), Point(y.x(),y.y(),y.z()));
          if (tree_.do_intersect(segment_query))
            z(i,3) = 0.0;
        }
        if( z(i,3) > 0.0 )
        {
          z(i,0) = std::ceil(pix(0));
          z(i,1) = std::ceil(pix(1));
        }
      }
      return z;
    }
    
    Eigen::MatrixXd senseWithNoise( const Eigen::Vector4d& x,
                                    const TargetModel &tmm )
    {
      Eigen::MatrixXd z = sense(x,tmm);
      // add noise
      for(int i = 0; i < z.rows(); ++i)
        if( z(i,3) > 0 && noise_stdev_ > 0.0 )
        {
          z(i,0) = std::ceil(std::max(std::min(z(i,0) + dis_(gen_),static_cast<double>(width_)-0.1), 0.1));
          z(i,1) = std::ceil(std::max(std::min(z(i,1) + dis_(gen_),static_cast<double>(height_)-0.1), 0.1));
        }
      return z;      
    }
                  
    void getJacobian( Eigen::MatrixXd &H,
                      Eigen::MatrixXd &V,
                      const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& p,
                      const Eigen::VectorXd& y,
                      int y_dim) const
    {
      V.setZero();
      H.setZero();
      int num_targets_known = static_cast<int>(y.rows()) / y_dim;
      for (int i = 0; i < num_targets_known; i++)
      {
        Eigen::Matrix<double, 2, 3> Jy = nx::cameraModelJacobianY( R, p, y.segment(y_dim * i, 3), K_);
        H.block(i, y_dim * i, 2, 3) = Jy;
        V.block(2 * i, 2 * i, 2, 2) = (Eigen::Matrix2d() << noise_stdev_*noise_stdev_, 0.0,
                                                            0.0, noise_stdev_*noise_stdev_).finished();
      }
    }

  };

}

#endif //INFO_GATHERING_CAMERAMODEL_H
