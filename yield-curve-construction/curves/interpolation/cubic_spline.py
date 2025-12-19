import numpy as np
from scipy.interpolate import CubicSpline

class CubicSplineLogDF:
    def fit(self, times, dfs):
        self.times = np.asarray(times, float)
        self.logdfs = np.log(np.asarray(dfs, float))
        self.cs = CubicSpline(self.times, self.logdfs, bc_type="natural", extrapolate=True)
        return self

    def df(self, t: float) -> float:
        return float(np.exp(self.cs(t)))
