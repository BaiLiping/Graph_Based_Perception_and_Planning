import yaml
import numpy as np

class MotionPrimitive:
    def __init__(self):
        self.uVec = []
        self.tVec = []
        self.cVec = []
        self.xVecVec = []  # micro states for collision checking

    def to_yaml(self):
        return {
            "controls": self.uVec,
            "durations": self.tVec,
            "costs": self.cVec
        }

    @classmethod
    def from_yaml(cls, data):
        mp = cls()
        mp.uVec = data["controls"]
        mp.tVec = data["durations"]
        mp.cVec = data["costs"]
        return mp

def mprms_from_yaml(from_yaml, f, x0, mprms, samp):
    with open(from_yaml) as file:
        docs = yaml.load_all(file, Loader=yaml.FullLoader)

    for doc in docs:
        mp = MotionPrimitive.from_yaml(doc)
        mprms.append(mp)
        
        np.cumsum(mp.cVec, out=mp.cVec)  # Cost should be cumulative (!)

        # Initialize the micro states using the dynamics f
        mp.xVecVec = [[] for _ in range(len(mp.uVec))]
        dt = 0.05
        x0_ = x0
        for u in range(len(mp.uVec)):
            tf = min(samp, mp.tVec[u])  # Use min of sampling time, or duration
            t = dt
            while t <= tf:
                mp.xVecVec[u].append(f(x0_, mp.uVec[u], t))
                t += dt
            x0_ = mp.xVecVec[u][-1]
