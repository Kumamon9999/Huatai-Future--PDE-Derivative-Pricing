import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
class Concentrate_Mesher:
    def __init__(self, start, end, size, cPoints, tol=1e-7):
        assert end > start, "end must be larger than start"
        
        self.start = start
        self.end = end
        self.size = size
        self.tol = tol
        
        self.points = np.array([cp[0] for cp in cPoints])
        self.betas = np.array([(cp[1] * (end - start)) ** 2 for cp in cPoints])
        self.required_points = np.array([cp[2] for cp in cPoints], dtype=bool)
        
        # Get initial scaling factor aInit
        self.aInit = self._get_initial_scaling_factor()
        
        # Adjust scaling factor a using Brent's method
        self.a = self._adjust_scaling_factor()
        
        # Solve ODE for all grid points
        self.locations = self._solve_ode_for_grid()
        
        # Ensure required points are part of the grid
        self._ensure_required_points()
        
        # Compute differences
        self.d_plus = np.diff(self.locations)
        self.d_minus = np.diff(self.locations)
        self.d_plus = np.append(self.d_plus, np.nan)
        self.d_minus = np.insert(self.d_minus, 0, np.nan)
    
    def _get_initial_scaling_factor(self):
        aInit = 0.0
        for point, beta in zip(self.points, self.betas):
            c1 = np.arcsinh((self.start - point) / beta)
            c2 = np.arcsinh((self.end - point) / beta)
            aInit += (c2 - c1) / len(self.points)
        return aInit
    
    def _adjust_scaling_factor(self):
        def objective(a):
            return self._solve_ode(a, self.start, 0.0, 1.0)[-1] - self.end
        
        # Check the behavior of the objective function
        a_min = 0.01 * self.aInit
        a_max = 10 * self.aInit
        fa_min = objective(a_min)
        fa_max = objective(a_max)
        
        if fa_min * fa_max > 0:
            raise ValueError(f"Objective function has the same sign at both ends of the interval "
                             f"[{a_min}, {a_max}]. fa_min = {fa_min}, fa_max = {fa_max}. "
                             f"Please check the initial interval or the encryption points.")
        
        return brentq(objective, a_min, a_max, rtol=self.tol)
    
    def _solve_ode(self, a, start, x0, x1):
        def ode(x, y):
            return a / np.sqrt(np.sum(1 / (self.betas + (y - self.points) ** 2)))
        
        sol = solve_ivp(ode, [x0, x1], [start], method='DOP853', t_eval=np.linspace(x0, x1, self.size), rtol=1e-8, atol=1e-8)
        return sol.y[0]
    
    def _solve_ode_for_grid(self):
        return self._solve_ode(self.a, self.start, 0.0, 1.0)
    
    def _ensure_required_points(self):
        x = np.linspace(0, 1, self.size)
        ode_solution = interp1d(x, self.locations, kind='linear', fill_value="extrapolate")
        
        w = [(0.0, 0.0)]
        for point, required in zip(self.points, self.required_points):
            if required and self.start < point < self.end:
                j = np.searchsorted(self.locations, point)
                e = brentq(lambda x: ode_solution(x) - point, x[j-1], x[j], rtol=self.tol)
                w.append((min(x[-2], x[j]), e))
        
        w.append((1.0, 1.0))
        w.sort()
        w = [(u, z) for u, z in w if not any(np.isclose(u, w_i[0]) for w_i in w if w_i != (u, z))]
        
        u, z = zip(*w)
        transform = interp1d(u, z, kind='linear', fill_value="extrapolate")
        
        self.locations = ode_solution(transform(np.linspace(0, 1, self.size)))
        self.locations.sort()
        np.unique(self.locations)

    def plot_grid_comparison(self):
        temp_grid=np.linspace(self.start,self.end,self.size)
        for u, x in zip(temp_grid, self.locations):
            plt.vlines(x=u, ymin=self.start, ymax=x, color='grey', linewidth=0.1)  # 调整垂直线的宽度
            plt.hlines(y=x, xmin=self.end, xmax=u, color='salmon', linewidth=0.1)  # 调整水平线的宽度
        # 调整主曲线的宽度
        plt.plot(temp_grid,self.locations, '-o', linewidth=0.1, markersize=0.3)  # 调整主曲线的宽度和标记大小
        plt.axis('square')
        plt.xlabel('Uniform Grid')
        plt.ylabel('Stretched Grid')
        plt.title('Composite')
        plt.show()
    