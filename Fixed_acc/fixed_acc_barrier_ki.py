import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
import sys
from Statistical_method.derivative import Derivative
from Statistical_method.Concentrate_grid import Concentrate_Mesher
from scipy.sparse import csr_matrix
class BasePricerAcc:

    def __init__(
            self,
            val_date,                               # 估值日期
            end_date,                               # 到期日
            spot,                                   # 当前标的价格
            vol,                                    # 波动率
            r,                                      # 无风险利率
            q,                                      # 红利
            barrier,
            strike,                                 # 执行价
            ki_barrier,                             # 敲入障碍价
            ki_strike,                              # 建仓价
            leverage,                               # 每日线性倍数
            final_position_quantity,                # 最后一日的线性倍数
            base_quantity,                          # 每日数量
            option_type,                            # 期权类型
            obs_price_list=None,                    # 已过观察日的收盘价格序列，默认为None，不输入则不考虑已派息部分value
            year_base=245,                          # 每年的天数， 交易日算一般定为245
            s_steps=2000,                           # S网格点切分的个数
            time_steps_per_day=1,                   # T网格点每日切分的个数，一般定为1
            fusing=False,                           # 敲出后是否熔断
            concentrate=True
    ):
        self.concentrate=concentrate
        self.val_date = val_date if val_date <= end_date else end_date
        self.intraday = (self.val_date - int(self.val_date) > 0)
        self.val_date_int = int(self.val_date) + 1 if self.intraday else self.val_date
        self.end_date = end_date
        self.spot = spot
        self.vol = vol
        self.r = r
        self.q = q
        self.s_steps=s_steps
        self.option_type = option_type
        self.base_quantity = base_quantity
        self.ki_barrier = ki_barrier
        self.ki_strike = ki_strike
        self.leverage = leverage
        self.final_position_quantity = final_position_quantity
        self.barrier=barrier
        self.strike = strike
        self.obs_price_list = obs_price_list
        self.option_type = option_type
        self.grid_ds=0
        self.year_base = year_base
        self.TimeStepsPerDay = time_steps_per_day
        self.fusing = fusing
        self.obs_value = 0
        self.s_min=0
        self.s_max=0
        self.calculate_min_max()
        self.ds = 0.01
        self.dv = 0.01
        self.dr = 0.0001
        self.dq = 0.001

        if self.option_type == 'call':
            self.sign = 1
        elif self.option_type == 'put':
            self.sign = -1
        else:
            raise Exception('Wrong Option Type!')
        self.S_grid, self.pv_matrix = self.calculate_pv_matrix()

    def calculate_min_max(self):
        sigma=3
        t_max = (self.end_date - self.val_date_int) / self.year_base
        coefficient=self.r-self.q-0.5*(self.vol**2)
        if coefficient>=0:
            self.s_min=np.exp(-coefficient*t_max-sigma*self.vol*np.sqrt(t_max))
            if coefficient>0 and sigma*self.vol/(2*coefficient)<np.sqrt(t_max):
                self.s_max=np.exp(sigma**2*self.vol**2/(4*coefficient))
            else:
                self.s_max=np.exp(max(0,-coefficient*t_max+sigma*self.vol*np.sqrt(t_max)))
        elif coefficient<0:
            self.s_max=np.exp(-coefficient*t_max+sigma*self.vol*np.sqrt(t_max))
            if -sigma*self.vol/(2*coefficient)<np.sqrt(t_max):
                self.s_min=np.exp(sigma**2*self.vol**2/(4*coefficient))
            else:
                self.s_min=np.exp(min(0,-coefficient*t_max-sigma*self.vol*np.sqrt(t_max)))
    
    def calculate_grid(self):
        t_max = (self.end_date - self.val_date_int) / self.year_base
        self.t_steps = int((self.end_date - self.val_date_int) * self.TimeStepsPerDay)
        self.dt = t_max / self.t_steps if self.t_steps > 0 else 0
        self.T_grid = np.array([i * self.dt for i in range(0, self.t_steps + 1, 1)])

        if hasattr(self, 'barrier'):
            if self.option_type=='call':
                s_max = max(self.s_max,1.05) * max(self.spot,self.barrier, self.strike)
                s_min=  min(self.s_min,0.95) * min(self.spot,self.ki_barrier)
            elif self.option_type=='put':
                s_max = max(self.s_max,1.05) * max(self.ki_barrier,self.spot)
                s_min=  min(self.s_min,0.95) * min(self.spot,self.barrier, self.strike)
            if self.concentrate:
                composite=Concentrate_Mesher(start=s_min,end=s_max,size=self.s_steps,
                                cPoints=[[self.strike,0.00001,True],[self.barrier,0.00001,True]])
                #composite.plot_grid_comparison()
                self.S_grid=composite.locations
            else:
                self.S_grid=np.linspace(s_min,s_max,self.s_steps+1)
        else:
            s_max = self.s_max * max(self.strike, self.spot)
            s_min=self.s_min*min(self.strike, self.spot)
            self.S_grid = np.linspace(s_min,s_max,self.s_steps+1)
        #print(self.S_grid)
    
    def lu_solve(self):
        J=self.S_grid[1:-1]
        D = csr_matrix(np.diag(J))
        a=Derivative(self.S_grid)
        self.Q = (self.dt*self.r+1)*sp.eye(len(self.S_grid) - 2)
        self.a=0.5*(self.vol*self.S_grid[0])**2*a.second_a+(self.r-self.q)*self.S_grid[0]*a.first_a
        self.c=0.5*(self.vol*self.S_grid[-1])**2*a.second_c+(self.r-self.q)*self.S_grid[-1]*a.first_c
        self.A=0.5*(self.vol*D)**2@a.s_second_derivative-(self.r-self.q)*D@a.s_first_derivative
        P=self.Q-self.dt*self.A
        self.Pinv = spilu(csc_matrix(P), drop_tol=1e-10, fill_factor=100)

    def ki_payment(self, spot, idx):
        quantity_a = self.base_quantity * self.leverage * (self.t_steps + 1 - idx)
        return self.sign * (spot - self.ki_strike) * (quantity_a + self.final_position_quantity)

    def final_position_payment(self, spot):
        return self.sign * (spot - self.ki_strike) * self.final_position_quantity

    def daily_payment(self, spot):
        return 0

    def set_matrix(self):

        T_grid = self.T_grid
        S_grid = self.S_grid
        self.s_steps=S_grid.size-1
        pv_matrix = np.zeros([self.s_steps + 1, self.t_steps + 1], dtype=float)
        ki_matrix = np.zeros([self.s_steps + 1, self.t_steps + 1], dtype=float)

        for i in range(0, self.t_steps + 1):
            ki_matrix[:, i] = (self.ki_payment(S_grid * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i])), i)
                               ) * np.exp(-self.r * (T_grid[-1] - T_grid[i]))

        # 终值条件
        for j in range(0, self.s_steps + 1):
            if self.sign * (S_grid[j] - self.ki_barrier) >= 0:
                pv_matrix[j, self.t_steps] = self.daily_payment(S_grid[j])
            else:
                pv_matrix[j, self.t_steps] = ki_matrix[j, self.t_steps]

        if self.option_type == 'call':

            # 下边界条件，敲入，从后一天开始累计，当天在计算网格的时候加总
            for i in range(0, self.t_steps):
                temp = 0
                for j in range(i + 1, self.t_steps + 1):
                    temp += self.daily_payment(S_grid[0]) * np.exp(-self.r * (T_grid[j] - T_grid[i]))
                pv_matrix[0, i] = temp + self.final_position_payment(S_grid[0] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i])))

            # 上边界条件，从后一天开始累计，当天在计算网格的时候加总
            for i in range(0, self.t_steps):
                temp = 0
                for j in range(i + 1, self.t_steps + 1):
                    temp += self.daily_payment(S_grid[-1] ) * np.exp(-self.r * (T_grid[j] - T_grid[i]))
                pv_matrix[-1, i] = temp

        else:

            # 上边界条件，敲入，从后一天开始累计，当天在计算网格的时候加总
            for i in range(0, self.t_steps):
                temp = 0
                for j in range(i + 1, self.t_steps + 1):
                    temp += self.daily_payment(S_grid[-1] ) * np.exp(-self.r * (T_grid[j] - T_grid[i]))
                pv_matrix[-1, i] = temp + self.final_position_payment(S_grid[-1] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i])))

            # 下边界条件，从后一天开始累计，当天在计算网格的时候加总
            for i in range(0, self.t_steps):
                temp = 0
                for j in range(i + 1, self.t_steps + 1):
                    temp += self.daily_payment(S_grid[0]) * np.exp(-self.r * (T_grid[j] - T_grid[i]))
                pv_matrix[0, i] = temp

        return pv_matrix, ki_matrix

    def calculate_pv_matrix(self):
        self.calculate_grid()
        S_grid = self.S_grid   
        pv_matrix, ki_matrix = self.set_matrix()
        self.lu_solve()

        ki_ids = np.where(self.sign * (S_grid - self.ki_barrier) < 0)
        if self.fusing and hasattr(self, 'barrier'):
            ko_ids = np.where(self.sign * (S_grid - self.barrier) >= 0)
        for i in range(self.t_steps - 1, -1, -1):
            pv_matrix[ki_ids, i + 1] = ki_matrix[ki_ids, i + 1].copy()
            Z = pv_matrix[1: -1, i + 1].copy()
            Z[0] += self.a * self.dt * pv_matrix[0, i]
            Z[-1] += self.c * self.dt * pv_matrix[-1, i]
            pv_matrix[1: -1, i] = self.Pinv.solve(Z)

            if self.fusing:
                pv_matrix[ko_ids, i] = np.array([self.daily_payment(s) for s in S_grid[ko_ids]]) * (self.t_steps - i)

            pv_matrix[:, i] += np.array([self.daily_payment(s) for s in S_grid])

        if self.intraday:
            dt = (self.val_date_int - self.val_date) / self.year_base
            P = self.Q - dt * self.A
            Pinv = spilu(csc_matrix(P), drop_tol=1e-10, fill_factor=100)

            Z = pv_matrix[1: -1, 0].copy()
            Z[0] += self.a * dt * pv_matrix[0, 0]
            Z[-1] += self.c * dt * pv_matrix[-1, 0]
            pv_matrix[1: -1, 0] = Pinv.solve(Z)

        return S_grid, pv_matrix[:, :2] + self.obs_value

    def get_pv(self, spot=None):
        spot = self.spot if spot is None else spot
        return np.interp(spot, self.S_grid, self.pv_matrix[:, 0])

