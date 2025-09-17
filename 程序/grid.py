from scipy.integrate import solve_ivp
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.sparse import diags
import scipy.sparse as sp
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
class CompositeTransformation:
    def __init__(self, x_star_all, a_all,s_min,s_max,s_steps):
        self.x_star_all = x_star_all
        self.a_all = a_all
        self.s_min=s_min
        self.s_max=s_max
        self.s_steps=s_steps
        self.u_grid=np.linspace(s_min,s_max,s_steps+1)
        self.du=(s_max-s_min)/s_steps
        self.A=None
        self.s_grid=None

    def __call__(self,t,x, A):
        total = 0.0
        for x_star, a in zip(self.x_star_all, self.a_all):
            total += np.power(a**2 + (x - x_star)**2, -2)
        return A * np.power(total, -0.5)

    def derivative(self,t, x, A): 
        # Manually calculate the derivative 
        delta = 1e-10 
        return (self.__call__(t,x + delta, A) - self.__call__(t,x - delta, A)) / (2 * delta)
    
    def __str__(self):
        retval = []
        for x_star, a in zip(self.x_star_all, self.a_all):
            retval.append(f'{x_star}/{a}')
        return ', '.join(retval)
    
    def compute_grid(self,A_try): 
        sol = solve_ivp(self, [self.s_min,self.s_max], [self.s_min], t_eval=self.u_grid, args=(A_try,)) 
        self.s_grid=sol.y[0]
        return self.s_grid
    
    def compute_A(self):
        result = root(lambda A: self.compute_grid(A)[-1] -self.s_max, [0],method='lm')
        assert result.success, result.message
        self.A = result.x[0]
        print(f'Optimal A: {self.A}')  # 打印最优的 A 值
        return self.A

    def compute_derivative(self): 
        derivatives = [] 
        for u, x in zip(self.s_grid, self.u_grid): 
            derivatives.append(self.derivative(u, x, self.A)) 
        return np.array(derivatives)
