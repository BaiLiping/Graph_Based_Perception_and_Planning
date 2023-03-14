import math
import numpy as np
import yaml
from pathlib import Path
from typing import List, Tuple

def meters2cells(m: float, minval: float, res: float) -> int:
    return int(math.floor((m - minval) / res))

def cells2meters(c: int, minval: float, res: float) -> float:
    return float(c * res + minval)

class MapSpec:
    def __init__(self, path_to_yaml: str):
        with open(path_to_yaml, 'r') as file:
            map_spec = yaml.safe_load(file)

        self.resolution = map_spec["resolution"]
        self.origin = map_spec["origin"]
        self.size_ = map_spec["size"]
        self.minval = [origin_i - 0.5 * res_i for origin_i, res_i in zip(self.origin, self.resolution)]
        
        if len(self.size_) > 0:
            yaml_parent_path = Path(path_to_yaml).parent
            with open(yaml_parent_path / map_spec["mappath"], 'r') as file:
                self.map = [float(val) for val in file.read().split()]
                self.map = np.array(self.map).reshape(self.size_)

def bresenham_step(dv: float, sv: float, svc: int, vmin: float, vres: float) -> Tuple[int, float, float]:
    if dv > 0:
        step_v = 1
        t_delta_v = vres / dv
        t_max_v = (vmin + (svc + 1) * vres - sv) / dv
    elif dv < 0:
        step_v = -1
        t_delta_v = vres / -dv
        t_max_v = (vmin + svc * vres - sv) / dv
    else:
        step_v = 0
        t_delta_v = 0.0
        t_max_v = float('inf')
    return step_v, t_delta_v, t_max_v

def bresenham2d(sx: float, sy: float, ex: float, ey: float,
                xmin: float, ymin: float, xres: float, yres: float) -> Tuple[List[int], List[int]]:
    xvec = []
    yvec = []

    sxc = meters2cells(sx, xmin, xres)
    exc = meters2cells(ex, xmin, xres)
    syc = meters2cells(sy, ymin, yres)
    eyc = meters2cells(ey, ymin, yres)

    dx = ex - sx
    dy = ey - sy

    step_x, t_delta_x, t_max_x = bresenham_step(dx, sx, sxc, xmin, xres)
    step_y, t_delta_y, t_max_y = bresenham_step(dy, sy, syc, ymin, yres)

    xvec.append(sxc)
    yvec.append(syc)

    while abs(sxc - exc) > 1 or abs(syc - eyc) > 1:
        if t_max_x < t_max_y:
            sxc += step_x
            t_max_x += t_delta_x
        else:
            syc += step_y
            t_max_y += t_delta_y

        xvec.append(sxc)
        yvec.append(syc)

    return xvec, yvec

def bresenham3d(sx: float, sy: float, sz: float, ex:float, ey: float, ez: float,
                xmin: float, ymin: float, zmin: float, xres: float, yres: float, zres: float) -> Tuple[List[int], List[int], List[int]]:
    xvec = []
    yvec = []
    zvec = []

    sxc = meters2cells(sx, xmin, xres)
    exc = meters2cells(ex, xmin, xres)
    syc = meters2cells(sy, ymin, yres)
    eyc = meters2cells(ey, ymin, yres)
    szc = meters2cells(sz, zmin, zres)
    ezc = meters2cells(ez, zmin, zres)

    dx = ex - sx
    dy = ey - sy
    dz = ez - sz

    step_x, t_delta_x, t_max_x = bresenham_step(dx, sx, sxc, xmin, xres)
    step_y, t_delta_y, t_max_y = bresenham_step(dy, sy, syc, ymin, yres)
    step_z, t_delta_z, t_max_z = bresenham_step(dz, sz, szc, zmin, zres)

    xvec.append(sxc)
    yvec.append(syc)
    zvec.append(szc)

    while abs(sxc - exc) > 1 or abs(syc - eyc) > 1 or abs(szc - ezc) > 1:
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                sxc += step_x
                t_max_x += t_delta_x
            else:
                szc += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                syc += step_y
                t_max_y += t_delta_y
            else:
                szc += step_z
                t_max_z += t_delta_z

        xvec.append(sxc)
        yvec.append(syc)
        zvec.append(szc)

    return xvec, yvec, zvec

# Example usage:

map_spec = MapSpec("path/to/your/yaml/file.yaml")
sx, sy, sz = 0.0, 0.0, 0.0
ex, ey, ez = 1.0, 1.0, 1.0
xvec, yvec, zvec = bresenham3d(sx, sy, sz, ex, ey, ez, *map_spec.minval, *map_spec.resolution)

print("X vector:", xvec)
print("Y vector:", yvec)
print("Z vector:", zvec)

