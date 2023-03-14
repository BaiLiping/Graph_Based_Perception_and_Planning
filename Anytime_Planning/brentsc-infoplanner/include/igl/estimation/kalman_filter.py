import numpy as np

class GaussianBelief:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

class KalmanFilter:
    @staticmethod
    def kf_covariance(cov_prior, A, W, H, V):
        # Predict
        cov_predict = A @ cov_prior @ A.T + W

        # Update
        R = H @ cov_predict @ H.T + V
        K = cov_predict @ H.T @ np.linalg.inv(R)  # Kalman Gain
        C = np.eye(cov_predict.shape[0]) - K @ H
        cov_update = C @ cov_predict

        # Return Result
        return cov_update

    @staticmethod
    def kf(mean_prior, cov_prior, A, W, H, V, innovation, debug=0):
        # Predict
        mean_predict = A @ mean_prior
        cov_predict = A @ cov_prior @ A.T + W

        # Update
        R = H @ cov_predict @ H.T + V
        K = cov_predict @ H.T @ np.linalg.inv(R)  # Kalman Gain
        C = np.eye(cov_predict.shape[0]) - K @ H
        cov_update = C @ cov_predict
        mean_update = mean_predict + K @ innovation

        if debug:
            KalmanFilter.print_debug(innovation, mean_prior, cov_prior, mean_predict, cov_predict, A, W, H, V, K, R, mean_update, cov_update)

        # Return Result
        return GaussianBelief(mean_update, cov_update)

    @staticmethod
    def print_debug(innovation, mean_prior, cov_prior, mean_predict, cov_predict, A, W, H, V, K, R, mean_update, cov_update):
        # Debug
        print("Kalman Gain: \n", K)
        print("R Mat: \n", R)
        print("H Mat: \n", H)
        print("V Mat: \n", V)
        print("A Mat: \n", A)
        print("W Mat: \n", W)
        print("Mean prior: \n", mean_prior)
        print("Sigma prior: \n", cov_prior)
        print("Mean pred: \n", mean_predict)
        print("Sigma pred: \n", cov_predict)
        print("Innovation (z-h_xy) = \n", innovation)
        print("Mean update: \n", mean_update)
        print("Sigma update: \n", cov_update)
