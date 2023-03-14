import numpy as np

def dd_motion_model_x(v, w, t):
    tw_ = t * w
    if abs(tw_) < 0.0001:
        return t * v * np.cos(tw_)
    else:
        return v / w * np.sin(tw_)

def dd_motion_model_y(v, w, t):
    tw_ = t * w
    if abs(tw_) < 0.0001:
        return t * v * np.sin(tw_)
    else:
        return v / w * (1 - np.cos(tw_))

def dd_motion_model_q(v, w, t):
    return t * w

def dd_motion_model(v, w, t):
    x = [0.0, 0.0, 0.0]
    x[2] = t * w
    if abs(x[2]) < 0.0001:
        x[0] = t * v * np.cos(x[2])
        x[1] = t * v * np.sin(x[2])
    else:
        x[0] = v / w * np.sin(x[2])
        x[1] = v / w * (1 - np.cos(x[2]))
    return x

def dd_motion_model_from_x_u(x, u, t):
    nx = [0.0, 0.0, 0.0]
    tw_ = t * u[1]
    nx[2] = np.remainder(x[2] + tw_, 2 * np.pi)
    if abs(tw_) < 0.0001:
        nx[0] = x[0] + t * u[0] * np.cos(nx[2])
        nx[1] = x[1] + t * u[0] * np.sin(nx[2])
    else:
        nx[0] = x[0] + u[0] / u[1] * (np.sin(nx[2]) - np.sin(x[2]))
        nx[1] = x[1] + u[0] / u[1] * (np.cos(x[2]) - np.cos(nx[2]))
    return nx

def polyval_nx(coeff, t):
    N = len(coeff) - 1
    if N < 0:
        return 0

    y = coeff[N]
    for c in range(N - 1, -1, -1):
        y = y * t + coeff[c]
    return y
