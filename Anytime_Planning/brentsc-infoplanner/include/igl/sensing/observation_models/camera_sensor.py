import numpy as np
from typing import Tuple
import open3d as o3d
from sensor import Sensor
from scipy.spatial.transform import Rotation as R

class CameraSensor(Sensor):
    def __init__(self, height: int, width: int, K: np.ndarray, noise_stdev: float, off_file: str = ""):
        super().__init__(2)
        self.height = height
        self.width = width
        self.K = K
        self.noise_stdev = noise_stdev
        self.mapSet = len(off_file) > 0
        
        if self.mapSet:
            self.mesh = o3d.io.read_triangle_mesh(off_file)
            
        self.gen = np.random.default_rng()
    
    def set_map(self, off_file_name: str):
        self.mesh = o3d.io.read_triangle_mesh(off_file_name)
        if self.mesh.has_triangles():
            self.mapSet = True

    def is_valid(self, z: np.ndarray) -> bool:
        return 0.0 < z[0] < self.width and 0.0 < z[1] < self.height

    def sense(self, x: np.ndarray, tmm) -> np.ndarray:
        p = np.array([x[0], x[1], x[3]])
        Rz = R.from_rotvec(np.array([0, 0, x[2]])).as_matrix()

        y_true = tmm.get_target_state()
        y_dim = tmm.dim
        num_targets = y_true.size // y_dim
        
        z = np.zeros((num_targets, 4))
        da = tmm.mgr.da_reverse
        for i in range(num_targets):
            y = y_true[y_dim * i : y_dim * (i + 1)]
            pix = camera_model(Rz, p, y, self.K)
            z[i, 2] = da[i]
            z[i, 3] = float(self.is_valid(pix))
            if self.mapSet and z[i, 3] > 0.0:
                segment_query = o3d.geometry.LineSegment(
                    o3d.geometry.Vector3d(p[0], p[1], p[2]), o3d.geometry.Vector3d(y[0], y[1], y[2])
                )
                if self.mesh.segment_intersects(segment_query):
                    z[i, 3] = 0.0
            if z[i, 3] > 0.0:
                z[i, 0] = np.ceil(pix[0])
                z[i, 1] = np.ceil(pix[1])
        return z

    def sense_with_noise(self, x: np.ndarray, tmm) -> np.ndarray:
        z = self.sense(x, tmm)
        for i in range(z.shape[0]):
            if z[i, 3] > 0 and self.noise_stdev > 0.0:
                z[i, 0] = np.ceil(np.clip(z[i, 0] + self.gen.normal(0, self.noise_stdev), 0.1, self.width - 0.1))
                z[i, 1] = np.ceil(np.clip(z[i, 1] + self.gen.normal(0, self.noise_stdev), 0.1, self.height - 0.1))
        return z
