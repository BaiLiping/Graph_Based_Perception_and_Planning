//
// Created by brent on 11/28/18.
//

#define CATCH_CONFIG_MAIN

#include "catch2/catch.hpp"

#include <igl/target_motion_models/info_target_model.h>
#include <Eigen/Dense>

nx::map_nd map;
 std::vector<char> cmap(0);

TEST_CASE( "Info Target Model Test", "[InfoInfoTargetModel]")
{
  std::vector<nx::InfoTarget> test_targets;
  // Create Vector of Targets to Test
  // Common State and Noise matrices
  MatrixXd A = MatrixXd::Identity(2, 2);
  MatrixXd W = 0.25 * MatrixXd::Identity(2, 2);
  MatrixXd covariance = 2 * MatrixXd::Identity(2,2);

  for (int i = 0; i < 5; i++) {
    Vector2d y;
    y << i, i;
    test_targets.push_back(nx::InfoTarget(nx::Target(i, y, A, W), covariance));
  }

  SECTION("Testing Empty Info Target Model") {
    nx::InfoTargetModel tmm(map, cmap);
    REQUIRE(tmm.getTargetState().rows() == 0);
    REQUIRE(tmm.getSystemMatrix().rows() == 0);
    REQUIRE(tmm.getNoiseMatrix().rows() == 0);
    REQUIRE(tmm.getCovarianceMatrix().rows() == 0);
  }

  SECTION("Test Adding Single Target") {
    nx::InfoTargetModel tmm(map, cmap);
    tmm.addTarget(0, test_targets[0]);

    REQUIRE(tmm.getTargetState().rows() == 2);
    REQUIRE(tmm.getSystemMatrix().rows() == 2);
    REQUIRE(tmm.getNoiseMatrix().rows() == 2);
    REQUIRE(tmm.getCovarianceMatrix().rows() == 2);


    Vector2d output;
    output << 0, 0;
    Matrix2d A_out = MatrixXd::Identity(2, 2);
    Matrix2d W_out = 0.25 * MatrixXd::Identity(2, 2);
    Matrix2d Sigma_out = 2 * MatrixXd::Identity(2, 2);

    REQUIRE(tmm.getTargetState() == output);
    REQUIRE(tmm.getSystemMatrix() == A_out);
    REQUIRE(tmm.getNoiseMatrix() == W_out);
    REQUIRE(tmm.getCovarianceMatrix() == Sigma_out);

  }

  SECTION("Test Multiple Targets and Removal") {
    nx::InfoTargetModel tmm(map, cmap);
    for (int i = 0; i < test_targets.size(); i++)
      tmm.addTarget(i, test_targets[i]);

    REQUIRE(tmm.getTargetState().rows() == 10);
    REQUIRE(tmm.getSystemMatrix().rows() == 10);
    REQUIRE(tmm.getNoiseMatrix().rows() == 10);
    REQUIRE(tmm.getCovarianceMatrix().rows() == 10);

    VectorXd output(10);
    output << 0, 0, 1, 1, 2, 2, 3, 3, 4, 4;
    MatrixXd A_out = MatrixXd::Identity(10, 10);
    MatrixXd W_out = 0.25 * MatrixXd::Identity(10, 10);
    MatrixXd Sigma_out = 2 * MatrixXd::Identity(10, 10);

    REQUIRE(tmm.getTargetState() == output);
    REQUIRE(tmm.getSystemMatrix() ==  A_out);
    REQUIRE(tmm.getNoiseMatrix() == W_out);
    REQUIRE(tmm.getCovarianceMatrix() == Sigma_out);

    tmm.removeTarget(3);
    VectorXd output_new(8);
    output_new << 0, 0, 1, 1, 2, 2, 4, 4;
    REQUIRE(tmm.getTargetState() == output_new);
    REQUIRE(tmm.getTargetState().rows() == 8);
    REQUIRE(tmm.getSystemMatrix().rows() == 8);
    REQUIRE(tmm.getNoiseMatrix().rows() == 8);
    REQUIRE(tmm.getCovarianceMatrix().rows() == 8);
  }

  SECTION("Tests the update_state function for consistency. ") {
    nx::InfoTargetModel tmm(map, cmap);
    for (int i = 0; i < test_targets.size(); i++)
      tmm.addTarget(i, test_targets[i]);

    REQUIRE(tmm.num_targets() == 5);
    // Update Target State.
    Vector2d new_state;
    new_state << -5, -5;
    tmm.updateTarget(0, new_state);
    REQUIRE(tmm.targets[0]->getPosition().head(2) == new_state);
  }

}