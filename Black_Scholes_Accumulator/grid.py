from scipy.integrate import solve_ivp
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
class CompositeTransformation:
    #u是均匀网格，对均匀网格做变化得到S不均匀网格
    def __init__(self, x_star_all, a_all,s_min,s_max,s_steps):
        self.x_star_all = x_star_all
        self.a_all = a_all
        self.s_min=s_min
        self.s_max=s_max
        self.s_steps=s_steps
        self.u_grid=np.linspace(s_min,s_max,s_steps+1)
        self.du=(s_max-s_min)/s_steps
        self.A=self.compute_A()
        self.s_grid=self.compute_grid(self.A,)
        self.first_orders=self.__call__(self.u_grid,self.s_grid,self.A)
        self.second_orders = self.compute_derivative()

    #计算导数dS/du
    def __call__(self,t,x, A):
        total = 0.0
        for x_star, a in zip(self.x_star_all, self.a_all):
            total += np.power(a**2 + (x - x_star)**2, -2)
        return A * np.power(total, -0.5)

    #计算二阶导数d^2 S/du^2
    def derivative(self,t, x, A): 
        # Manually calculate the derivative 
        delta = 1e-10 
        return (self.__call__(t,x + delta, A) - self.__call__(t,x - delta, A)) / (2 * delta)
    
    #得到S的网格
    def compute_grid(self,A_try): 
        sol = solve_ivp(self, [self.s_min,self.s_max], [self.s_min], t_eval=self.u_grid, args=(A_try,)) 
        self.s_grid=sol.y[0]
        return self.s_grid
    
    #计算满足两个边界条件的ODE的参数A
    def compute_A(self):
        result = root(lambda A: self.compute_grid(A)[-1] -self.s_max, [0],method='lm')
        assert result.success, result.message
        self.A = result.x[0]
        return self.A

    #计算S网格的导数
    def compute_derivative(self): 
        derivatives = [] 
        for u, x in zip(self.s_grid, self.u_grid): 
            derivatives.append(self.derivative(u, x, self.A)) 
        return np.array(derivatives)
    
    #画图，S网格和均匀网格
    def plt_grid(self):
        for u, x in zip(self.u_grid, self.s_grid):
            plt.vlines(x=u, ymin=0, ymax=x, color='grey', linewidth=0.1)  # 调整垂直线的宽度
            plt.hlines(y=x, xmin=0, xmax=u, color='salmon', linewidth=0.1)  # 调整水平线的宽度
        # 调整主曲线的宽度
        plt.plot(self.u_grid, self.s_grid, '-o', linewidth=0.1, markersize=0.3)  # 调整主曲线的宽度和标记大小
        plt.axis('square')
        plt.xlabel('Uniform Grid')
        plt.ylabel('Stretched Grid')
        plt.title('Composite')
        plt.show()