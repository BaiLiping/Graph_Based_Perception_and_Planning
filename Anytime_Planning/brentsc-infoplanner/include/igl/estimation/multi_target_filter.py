import numpy as np
from typing import List, TypeVar, Generic, Any

State = TypeVar('State')
Measurement = Any  # Replace this with the appropriate Measurement class/type

class MultiTargetFilter:
    @staticmethod
    def multi_target_kf_covariance(robot, x_t, y_t, cov_prior):
        # Compute dimensions of the problem.
        num_targets_known = robot.tmm.num_targets()
        y_dim = robot.tmm.target_dim // num_targets_known
        z_dim = robot.sensor.z_dim
        # Allocate matrices.
        A = np.zeros((num_targets_known * y_dim, num_targets_known * y_dim))
        W = np.zeros((num_targets_known * y_dim, num_targets_known * y_dim))
        H = np.zeros((num_targets_known * z_dim, num_targets_known * y_dim))
        V = np.zeros((num_targets_known * z_dim, num_targets_known * z_dim))
        # Get system and observation matrices from robot's properties.
        robot.tmm.get_jacobian(A, W)
        robot.sensor.get_jacobian(H, V, x_t, robot.tmm, y_t)

        return KalmanFilter.kf_covariance(cov_prior, A, W, H, V)

    @staticmethod
    def multi_target_kf(measurements, robot, debug=False):
        x_t = robot.get_state()
        mean_prior = robot.tmm.get_target_state()
        cov_prior = robot.tmm.get_covariance_matrix()
        # Get problem dimension.
        num_targets = robot.tmm.num_targets()
        y_dim = robot.tmm.target_dim // num_targets
        z_dim = robot.sensor.z_dim
        # Allocate matrices.
        A = np.zeros((num_targets * y_dim, num_targets * y_dim))
        W = np.zeros((num_targets * y_dim, num_targets * y_dim))
        H = np.zeros((num_targets * z_dim, num_targets * y_dim))
        V = np.zeros((num_targets * z_dim, num_targets * z_dim))

        # Get Target Motion Model, and Sensor Observation Model
        validity = [m.valid for m in measurements]
        robot.tmm.get_jacobian(A, W)

        innovation = np.zeros(z_dim * num_targets)
        H.fill(0)
        V.fill(0)
        for i, measurement in enumerate(measurements):
            target = robot.tmm.get_target_by_id(measurement.ID)
            y_predict = target.predict_state(1)
            # Compute Jacobian entries only for valid measurements
            H_i = np.zeros((z_dim, y_dim))
            V_i = np.zeros((z_dim, z_dim))
            robot.sensor.get_jacobian(H_i, V_i, x_t, y_predict, False)  # Don't check validity again.
            if validity[i]:
                H[i * z_dim:(i + 1) * z_dim, i * y_dim:(i + 1) * y_dim] = H_i
            V[i * z_dim:(i + 1) * z_dim, i * z_dim:(i + 1) * z_dim] = V_i

            # Compute Innovation
            z = measurement.z
            h_xy = robot.sensor.observation_model(x_t, y_predict)
            innovation[i * z_dim:(i + 1) * z_dim] = robot.sensor.compute_innovation(z, h_xy)

            if debug:
                print(f"Target: id={target.ID} state={target.get_position()}")
                print(f"Measurement: {z}")
                print(f"H_XY: {h_xy}")

        result = KalmanFilter.kf(mean_prior, cov_prior, A, W, H, V, innovation, debug)
        return GaussianBelief(result.mean, result.cov)
