
import numpy as np
import numpy.linalg as LA


def wrapAngle(theta):
    theta += np.pi
    theta = np.fmod(theta, 2 * np.pi)
    theta -= np.pi
    return theta

def getJacobian(x, y):
    range, bearing = getObservation(x, y)
    H = np.zeros((2, 4))
    H[0, 0] = y[0] - x[0]
    H[0, 1] = y[1] - x[1]
    H[1, 0] = -np.sin(bearing + x[2])
    H[1, 1] = np.cos(bearing + x[2])
    H = H / range
    return H

def getObservation(x, y):
    range = LA.norm((x[0:2] - y[0:2]))
    bearing = wrapAngle(np.arctan2(y[0] - x[0], y[1] - x[1]) - x[2])
    return range, bearing

def sense(x, y):
    range, bearing = getObservation(x,y)
    range += np.random.normal(0, 0.1)
    return range, bearing


# Script to test Kalman Filter for range-bearing sensors.
tau = 0.5 # Time discretization
q = 0.1  # Diffusion / Noise Parameter
A = np.eye((4))
I2 = np.identity(2)
A[0:2, 2:4] = tau * I2
W = q * np.ones((4, 4))
W[0:2, 0:2] *= tau ** 3 / 3
W[0:2, 2:4] *= tau ** 2 / 2
W[2:4, 0:2] *= tau ** 2 / 2
W[2:4, 2:4] *= tau

V = np.array([[0.1, 0], [0, 0.05]])

robot_states = [[0, 0, 0], [0, 0, np.pi], [0, -1, np.pi/2], [0, 1, np.pi/3],
                [0, 0, 0], [0, 0, np.pi], [0, -1, np.pi/2], [0, 1, np.pi/3],
                [0, 0, 0], [0, 0, np.pi], [0, -1, np.pi/2], [0, 1, np.pi/3],
                [0, 0, 0], [0, 0, np.pi], [0, -1, np.pi/2], [0, 1, np.pi/3],
                [0, 0, 0], [0, 0, np.pi], [0, -1, np.pi/2], [0, 1, np.pi/3]]
# Ground Truth Target information.
y_true = np.array([-4, 0, 0, 0])
# Estimate information
y_est = y_true
Sigma = np.eye(4) * 2
t = 0
for state in robot_states:
    y_prior, Sigma_prior = y_est, Sigma # Capture Prior
    t += 1
    range, bearing = sense(state, y_true)      # Sense
    z_pred, bearing_pred = getObservation(state, y_est)     # Estimate
    innovation = np.array([range - z_pred, bearing - bearing_pred])     # Predict measurement

    # Kalman Filter (Estimate)
    y_pred = np.dot(A,  y_est) # Predict
    Sigma_pred = np.dot(np.dot(A , Sigma),  A.T) + W
    H = getJacobian(state, np.dot(A, y_est))     # Update
    R = np.dot(np.dot(H, Sigma_pred), H.T) + V
    K = np.dot(np.dot(Sigma_pred , H.T), np.linalg.inv(R))
    C = np.eye(4, 4) - np.dot(K, H)

    Sigma = np.dot(C,  Sigma)
    y_est = y_est + np.dot(K, innovation)



    y_true = np.dot(A, y_true) + np.random.multivariate_normal([0,0,0,0], W.tolist())

    print(f"Problem Setup: ",
          "Sensing from", state, "\n",
          "A=", A, "\n",
           "W=", W, "\n",
          "V=", V, "\n",
          "H=", H, "\n",
          "K=", K, "\n")

    print("Prior Distribution: ", y_prior, Sigma_prior)
    print("Predicted Distribution: ", y_pred, Sigma_pred)
    print("Posterior Distribution ", y_est, Sigma)
    print("Ground Truth ", y_true)
    print("timestep: ", t)