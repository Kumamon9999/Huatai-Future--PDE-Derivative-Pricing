import numpy as np
import scipy.sparse as sp
class Derivative:
    def __init__(
            self,
            s_grid,
            s_minus_grid=None,
            s_plus_grid=None,
            s_first_derivative=None,
            s_second_derivative=None,
        ):
        self.s_grid=s_grid
        self.s_minus_grid=self.minus_grid()
        self.s_plus_grid=self.plus_grid()
        self.s_first_derivative=self.first_derivative()
        self.s_second_derivative=self.second_derivative()

    def minus_grid(self):
        #计算 S_i - S_{i-1}，minus[0] = 0
        minus = np.zeros_like(self.s_grid)
        minus[1:] = self.s_grid[1:] - self.s_grid[:-1]
        return minus

    def plus_grid(self):
        #计算 S_{i+1} - S_i，plus[-1] = 0
        plus = np.zeros_like(self.s_grid)
        plus[:-1] = self.s_grid[1:] - self.s_grid[:-1]
        return plus

    def first_derivative(self):
        hm=self.s_minus_grid
        hp=self.s_plus_grid
        zetam1 = hm*(hm+hp);
        zeta0  = hm*hp;
        zetap1 = hp*(hm+hp);
        lower=np.zeros(self.s_grid.size-2)
        diag=np.zeros(self.s_grid.size-2)
        upper=np.zeros(self.s_grid.size-2)
        for i in range(len(lower)-2):
            lower[i] = -hp[i+1] / zetam1[i+1]
            diag[i] = (hp[i+1] - hm[i+1]) / zeta0[i+1]
            upper[i] = hm [i+1]/ zetap1[i+1]
        A=sp.diags([lower[1:], diag,upper[:-1]], [-1, 0, 1], format='csc')
        self.first_a=lower[0]
        self.first_c=upper[-1]
        return A

    def second_derivative(self):
        hm=self.s_minus_grid
        hp=self.s_plus_grid
        zetam1 = hm*(hm+hp);
        zeta0  = hm*hp;
        zetap1 = hp*(hm+hp);
        lower=np.zeros(self.s_grid.size-2)
        diag=np.zeros(self.s_grid.size-2)
        upper=np.zeros(self.s_grid.size-2)
        for i in range(len(lower)-2):
            lower[i] = 2/zetam1[i+1]
            diag[i] = -2/zeta0[i+1]
            upper[i] = 2/zetap1[i+1]
        B=sp.diags([lower[1:], diag,upper[:-1]], [-1, 0, 1], format='csc')
        self.second_a=lower[0]
        self.second_c=upper[-1]
        return B