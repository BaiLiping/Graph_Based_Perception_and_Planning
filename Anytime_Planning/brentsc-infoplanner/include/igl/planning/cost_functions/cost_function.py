import numpy as np

class CostFunction:
    def compute_cost(self, Sigma, t=0):
        return np.log(np.linalg.det(Sigma))