import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
import sys
from Statistical_method.derivative import Derivative
from Statistical_method.Concentrate_grid import Concentrate_Mesher
from scipy.sparse import csr_matrix
class Snowball:

    def __init__(
            self,
            val_date,                               # 估值日期
            end_date,                               # 到期日
            notional,                                #nominal capital
            spot,                                   # 当前标的价格
            s0,                                     #entering price
            vol,                                    # 波动率
            r,                                      # 无风险利率
            q,                                      # 红利
            strike,                                 # 执行价
            ko_coupon,                              # coupon for knocking out
            no_ko_coupon,                           # coupons(dividend) for not knocking out
            ko_barrier,                             # knock_out障碍价
            #ko_obs_dates,                           # knock_out observation date
            option_type,                            # 期权类型
            year_base=245,                          # 每年的天数， 交易日算一般定为245
            s_steps=2000,                           # S网格点切分的个数
            time_steps_per_day=1,                   # T网格点每日切分的个数，一般定为1
            ki_flag=False,                                # whether knock_in
            concentrate=True
    ):
        self.concentrate=concentrate
        self.val_date = val_date if val_date <= end_date else end_date
        self.intraday = (self.val_date - int(self.val_date) > 0)
        self.val_date_int = int(self.val_date) + 1 if self.intraday else self.val_date
        self.end_date = end_date
        self.spot = spot
        self.notional=notional
        self.vol = vol
        self.r = r
        self.q = q
        self.option_type = option_type
        self.ko_barrier = ko_barrier
        self.no_ko_coupon=no_ko_coupon
        self.ki_flag=ki_flag
        self.ko_coupon=ko_coupon
        self.year_base = year_base
        self.daily_ko_coupon=self.ko_coupon/self.year_base
        self.daily_no_ko_coupon=self.no_ko_coupon/self.year_base
        self.s0=s0
        self.amount=self.notional/self.s0
        self.strike = strike
        #self.ko_obs_dates=ko_obs_dates
        self.option_type = option_type
        self.grid_ds=0
        self.TimeStepsPerDay = time_steps_per_day
        self.s_steps=s_steps
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
        self.S_grid, self.ki_matrix,self.pv_matrix = self.calculate_pv_matrix()

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
        if self.option_type=='call':
            s_max = max(self.s_max,1.05) * max(self.spot,self.ko_barrier, self.strike)
            s_min=  min(self.s_min,0.95) * min(self.spot,self.ko_barrier)
        elif self.option_type=='put':
            s_max = max(self.s_max,1.05) * max(self.ko_barrier,self.spot)
            s_min=  min(self.s_min,0.95) * min(self.spot,self.ko_barrier, self.strike)
        if self.concentrate:
            composite=Concentrate_Mesher(start=s_min,end=s_max,size=self.s_steps,
                            cPoints=[[self.strike,0.001,True],[self.ko_barrier,0.001,True]])
            #composite.plot_grid_comparison()
            self.S_grid=composite.locations
            self.grid_ds=(s_max-s_min)/self.s_steps
        else:
            print("111")
            self.S_grid=np.linspace(s_min,s_max,self.s_steps+1)
        print(self.S_grid)

    
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

    def ko_payment(self,spot,i):#i stands for the i^th day
        payment=self.ko_coupon*i/self.year_base*self.s0
        return payment
    
    def ki_payment(self,spot,i): #i stands for the i^th day
        if self.sign*(spot-self.ko_barrier)>=0:
            payment=self.ko_coupon*i/self.year_base*self.s0
        else:
            payment=spot
        return payment
    
    def final_payment(self,spot):
        if self.sign*(spot-self.ko_barrier)>=0:
            payment=self.ko_coupon*self.t_steps/self.year_base*self.s0
        elif self.sign*(spot-self.strike)>=0:
            payment=self.no_ko_coupon/self.year_base*self.s0
        else:
            payment=spot
        return payment
    
    def daily_payment(self,spot):
        return 0
    
    def set_matrix(self):
        T_grid = self.T_grid
        S_grid = self.S_grid
        self.s_steps=S_grid.size-1
        ki_matrix = np.zeros([self.s_steps + 1, self.t_steps + 1], dtype=float)
        pv_matrix = np.zeros([self.s_steps + 1, self.t_steps + 1], dtype=float)
        # 终值条件
        for j in range(0, self.s_steps + 1):
            ki_matrix[j, self.t_steps] = self.ki_payment(S_grid[j],self.t_steps)
            pv_matrix[j,self.t_steps]= self.final_payment(S_grid[j])

        if self.option_type == 'call':
            for i in range(self.t_steps+1):
            # upper bound condition， Knock out
                ki_matrix[-1,i] = self.ko_payment(S_grid[-1],i)
                pv_matrix[-1,i] = self.ko_payment(S_grid[-1],i)
            #lower bound condition, 
                ki_matrix[0,i] = S_grid[0] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i]))
                pv_matrix[0,i] = S_grid[0] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i]))
        else:
            for i in range(self.t_steps+1):
            # lower bound condition， Knock out
                ki_matrix[0,i] = self.ko_payment(S_grid[0],i)
                pv_matrix[0,i] = self.ko_payment(S_grid[0],i)
            #lower bound condition, 
                ki_matrix[-1,i] = S_grid[0] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i]))
                pv_matrix[-1,i] = S_grid[0] * np.exp((self.r - self.q) * (T_grid[-1] - T_grid[i]))
        return pv_matrix,ki_matrix


    def calculate_pv_matrix(self):
        self.calculate_grid()
        S_grid = self.S_grid
        pv_matrix,ki_matrix = self.set_matrix()
        self.lu_solve()

        ko_ids = np.where(self.sign * (S_grid - self.ko_barrier) >= 0)
        ki_ids = np.where(self.sign * (S_grid - self.strike) < 0)

        for i in range(self.t_steps - 1, -1, -1):

            #solve ki_matrix
            ki_matrix[ko_ids,i+1]= self.ko_payment(S_grid[ko_ids],i+1)
            Z = ki_matrix[1: -1, i + 1].copy()
            Z[0] += self.a * self.dt * ki_matrix[0, i]
            Z[-1] += self.c * self.dt * ki_matrix[-1, i]
            ki_matrix[1: -1, i] = self.Pinv.solve(Z)
            
            #solve pv_matrix
            pv_matrix[ki_ids, i + 1] = ki_matrix[ki_ids, i + 1].copy()
            pv_matrix[ko_ids, i + 1] = ki_matrix[ko_ids, i + 1].copy()
            W = pv_matrix[1: -1, i + 1].copy()
            W[0] += self.a * self.dt * pv_matrix[0, i]
            W[-1] += self.c * self.dt * pv_matrix[-1, i]
            pv_matrix[1: -1, i] = self.Pinv.solve(W)
            pv_matrix[:, i] += np.array([self.daily_payment(s) for s in S_grid])

        if self.intraday:
            dt = (self.val_date_int - self.val_date) / self.year_base
            P = self.Q - dt * self.A
            Pinv = spilu(csc_matrix(P), drop_tol=1e-10, fill_factor=100)
            Z = ki_matrix[1: -1, 0].copy()
            Z[0] += self.a * dt * ki_matrix[0, 0]
            Z[-1] += self.c * dt * ki_matrix[-1, 0]
            ki_matrix[1: -1, 0] = Pinv.solve(Z)

            W = pv_matrix[1: -1, 0].copy()
            W[0] += self.a * self.dt * pv_matrix[0, 0]
            W[-1] += self.c * self.dt * pv_matrix[-1,0]
            pv_matrix[1: -1, 0] = self.Pinv.solve(W)

        return S_grid, ki_matrix[:,:2],pv_matrix[:, :2]

    def get_pv(self, spot=None):
        spot = self.spot if spot is None else spot
        if self.ki_flag:
            return self.amount*np.interp(spot, self.S_grid, self.ki_matrix[:, 0])
        else:
            return self.amount*np.interp(spot, self.S_grid, self.pv_matrix[:, 0])