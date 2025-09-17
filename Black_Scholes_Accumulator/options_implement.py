from options_module import options
from SDE_class import SDE
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix
import math
from grid import CompositeTransformation
import matplotlib.pyplot as plt
#累计转远期类
class accumulator_forward(options):
    def __init__(self, option_type, spot, strike,barrier,ki_strike,ki_barrier, r, val_date,end_date, 
                 vol,payoff,base_quantity,leverage,final_position_quantity,q=0,year_base=245,s_steps=500,t_steps_daily=200):
        super().__init__(option_type, spot, strike, r, val_date,end_date,vol,q=0,year_base=year_base)
        
         #set maximum and minimum for grid，默认exp(+-2\sigma)
        self.s_max=np.exp(2*self.vol)*self.spot
        self.s_min=np.exp(-2*self.vol)*self.spot
        self.s_steps=s_steps

        #set the ununiform grid
        self.composite=CompositeTransformation(x_star_all=[ki_barrier,strike],a_all=[0.2*ki_barrier,0.2*strike],s_min=self.s_min,s_max=self.s_max,s_steps=self.s_steps)

        self.payoff=payoff
        self.base_quantity=base_quantity
        self.leverage=leverage
        self.final_position_quantity=final_position_quantity
        self.barrier=barrier
        self.ki_strike=ki_strike
        self.ki_barrier=ki_barrier
        self.t_day=end_date-val_date+1 #距离到期有几天
        self.t_year=self.t_day/self.year_base #距离到期有几年
        self.t_steps_daily=t_steps_daily
        self.s_uni_grid=np.linspace(self.s_min,self.s_max,self.s_steps+1) #均匀网格
        self.s_ununi_grid=self.composite.s_grid #不均匀网格
    
    def MC_method(self,sim_times):
        #向上取整，便于定义循环次数
        T_whole_day=math.ceil(self.t_day)
        dt=1/self.year_base
        price=0
        if self.option_type=='C':
            a=SDE.MC_matrix(self.spot,self.r,self.q,self.vol,sim_times,self.t_day,dt)
            #calculate the final payoff
            price=0
            for i in range(0,sim_times):
                final_payoff=0
                left_days=0
                knock_in=False
                for t in range(1, T_whole_day+1):
                    if a[i, t] >= self.ki_barrier:
                        if a[i, t] >= self.barrier or a[i, t] < self.strike:
                            continue
                        elif self.strike <= a[i, t] < self.barrier:
                            final_payoff+=self.payoff*self.base_quantity
                    else:
                        knock_in=True
                        left_days=T_whole_day-t+1
                        break
                if knock_in==True:
                    final_payoff+=(a[i,T_whole_day]-self.ki_strike)*(left_days*self.base_quantity*self.leverage+self.final_position_quantity)
                price+=np.exp(-self.r*self.t_year)*final_payoff/sim_times
        
        if self.option_type=='P':
            a=SDE.MC_matrix(self.spot,self.r,self.q,self.vol,sim_times,self.t_day,dt)
            #calculate the final payoff
            price=0
            for i in range(0,sim_times):
                final_payoff=0
                left_days=0
                knock_in=False
                for t in range(1, T_whole_day+1):
                    if a[i, t] <= self.ki_barrier:
                        if a[i, t] <= self.barrier or a[i, t] > self.strike:
                            continue
                        elif self.barrier < a[i, t] <=self.strike:
                            final_payoff+=self.payoff*self.base_quantity
                    else:
                        knock_in=True
                        left_days=T_whole_day-t+1
                        break
                if knock_in==True:
                    final_payoff+=(self.ki_strike-a[i,T_whole_day])*(left_days*self.base_quantity*self.leverage+self.final_position_quantity)
                price+=np.exp(-self.r*self.t_year)*final_payoff/sim_times
        return price
    
    #远期合约价格
    def forward_price(self,spot,strike,time_to_expire):
        return spot*np.exp(-self.q*time_to_expire)-strike*np.exp(-self.r*time_to_expire)
    
    #FDM解，解一日的，根据收盘价解开盘价
    def FDM_solver(self,pv_matrix,fdm_matrix):
        dt=1/(self.year_base*self.t_steps_daily)
        for i in range(self.t_steps_daily-1, -1, -1):
            Z = pv_matrix[1: -1, i + 1].copy()
            Z[0] += fdm_matrix.Aa[0] * dt * pv_matrix[0, i]
            Z[-1] += fdm_matrix.Ac[-1] * dt * pv_matrix[-1, i]
            pv_matrix[1: -1, i] = fdm_matrix.Pinv.solve(Z)
        return pv_matrix[:,0]

    #计算日内边界条件
    def calculate_pv_matrix(self,s_grid,entire_day_to_expire,t_day,last_price):
        pv_matrix = np.zeros((self.s_steps + 1, self.t_steps_daily + 1))
        if self.option_type=='C':
            for j in range(self.t_steps_daily + 1):
                #股价最低点，按当时的远期合约计算
                t_left = (entire_day_to_expire + t_day*(1 - j / self.t_steps_daily)) / self.year_base
                pv_matrix[0, j] = self.forward_price(self.s_min, self.ki_strike, t_left)*(self.base_quantity*self.leverage*(entire_day_to_expire+1)+self.final_position_quantity)
            pv_matrix[-1,:]=0
            for j in range(self.s_steps+1):
                #根据收盘观察价计算收盘的payoff，如果没敲入则要在原有价格加上payoff，敲入则直接变为远期合约
                pv_matrix[j,self.t_steps_daily]+=last_price[j] if s_grid[j]>self.ki_barrier else 0
                pv_matrix[j,self.t_steps_daily]+=self.payoff*self.base_quantity if s_grid[j]<self.barrier and s_grid[j]>=self.strike else 0
                #敲入，直接是远期合约价格
                if s_grid[j]<=self.ki_barrier:
                    pv_matrix[j,self.t_steps_daily]=self.forward_price(s_grid[j], self.ki_strike, entire_day_to_expire/self.year_base)*(self.base_quantity*self.leverage*(entire_day_to_expire+1)+self.final_position_quantity) 
        if self.option_type=='P':
            for j in range(self.t_steps_daily + 1):
                t_left = (entire_day_to_expire + t_day*(1 - j / self.t_steps_daily)) / self.year_base
                pv_matrix[-1, j] = -self.forward_price(self.s_max, self.ki_strike, t_left)*(self.base_quantity*self.leverage*(entire_day_to_expire+1)+self.final_position_quantity)
            pv_matrix[0,:]=0
            for j in range(self.s_steps+1):
                pv_matrix[j,self.t_steps_daily]+=last_price[j] if s_grid[j]<self.ki_barrier else 0
                pv_matrix[j,self.t_steps_daily]+=self.payoff*self.base_quantity if s_grid[j]>self.barrier and s_grid[j]<=self.strike else 0
                if s_grid[j]>=self.ki_barrier:
                    pv_matrix[j,self.t_steps_daily]=-self.forward_price(s_grid[j], self.ki_strike, entire_day_to_expire/self.year_base)*(self.base_quantity*self.leverage*(entire_day_to_expire+1)+self.final_position_quantity) 
        return pv_matrix
    
    #FDM解，往前循环解所有交易日的
    def FDM_pricing_uniform(self):
        entire_day_to_expire=math.ceil(self.t_day)
        decimal=1-(entire_day_to_expire-self.t_day)

        #last_price是lim t->t_i^+ price，没有加上payoff的次日开盘价
        last_price=np.zeros(self.s_steps+1)
        dt=1/(self.year_base*self.t_steps_daily)

        #系数矩阵
        fdm_matrix=FDM_matrix_uniform(self.vol,self.r,self.q,self.s_uni_grid,dt)

        for i in range(0,entire_day_to_expire-1):
            #将次日开盘价加上当日收盘支付
            pv_matrix=self.calculate_pv_matrix(self.s_uni_grid,i,1,last_price)
            #计算当日开盘价
            last_price=self.FDM_solver(pv_matrix,fdm_matrix)

        #小数部分处理，第一个交易日
        pv_matrix=self.calculate_pv_matrix(self.s_uni_grid,entire_day_to_expire-1,decimal,last_price)
        add_last_price=self.FDM_solver(pv_matrix,fdm_matrix)
        return add_last_price
    
    def FDM_pricing_ununiform(self):
        entire_day_to_expire=math.ceil(self.t_day)
        decimal=1-(entire_day_to_expire-self.t_day)
        last_price=np.zeros(self.s_steps+1)
        dt=1/(self.year_base*self.t_steps_daily)

        #此处调用需要用到composite的结构
        fdm_matrix=FDM_matrix_ununiform(self.vol,self.r,self.q,self.composite,dt)

        for i in range(0,entire_day_to_expire-1):
            pv_matrix=self.calculate_pv_matrix(self.s_ununi_grid,i,1,last_price)
            last_price=self.FDM_solver(pv_matrix,fdm_matrix)
        pv_matrix=self.calculate_pv_matrix(self.s_ununi_grid,entire_day_to_expire-1,decimal,last_price)
        add_last_price=self.FDM_solver(pv_matrix,fdm_matrix)
        return add_last_price

    #画图，画均匀网格和不均匀网格对比，只需要看颜色深浅
    def plt_grid(self):
        self.composite.plt_grid()
        return 


#均匀网格系数矩阵
class FDM_matrix_uniform():
    def __init__(self,vol,r,q,s_grid,dt):
        self.dt=dt
        self.vol=vol
        self.r=r
        self.q=q
        self.s_grid=s_grid
        self.ds=(s_grid[-1]-s_grid[0])/(len(s_grid)-1)
        self.Aa = 0.5 * ((self.vol*self.s_grid[1:-1]/self.ds) ** 2 - (self.r-self.q)*self.s_grid[1:-1]/self.ds)
        self.Ab = -(self.vol*self.s_grid[1:-1]/self.ds) ** 2 - self.r
        self.Ac = 0.5 * ((self.vol*self.s_grid[1:-1]/self.ds) ** 2 + (self.r-self.q)*self.s_grid[1:-1]/self.ds)
        self.A = sp.diags([self.Aa[1:], self.Ab, self.Ac[:-1]], [-1, 0, 1], format='csc')
        self.Q = sp.eye(len(s_grid)-2)
        self.P = self.Q - self.dt * self.A
        self.Pinv = spilu(csc_matrix(self.P), drop_tol=1e-10, fill_factor=100)

#不均匀网格系数矩阵，需要用到composite以及对应的属性如一阶导、二阶导等
class FDM_matrix_ununiform():
    def __init__(self,vol,r,q,composite,dt):
        self.vol=vol
        self.r=r
        self.q=q
        self.composite=composite
        self.dt=dt
        self.s_steps=len(composite.u_grid)-1
        du=(composite.u_grid[-1]-composite.u_grid[0])/self.s_steps
        a=(self.vol*self.composite.s_grid)**2/(2*self.composite.first_orders**2)
        b=(-self.composite.second_orders*(self.vol*self.composite.s_grid)**2)/(2*np.power(self.composite.first_orders,3))+(self.r-self.q)*self.composite.s_grid/self.composite.first_orders  
        self.Aa = a[1:self.s_steps]/(du**2)-b[1:self.s_steps]/(2*du)
        self.Ab = -2*a[1:self.s_steps]/(du**2)-self.r
        self.Ac = a[1:self.s_steps]/(du**2)+b[1:self.s_steps]/(2*du)
        self.A = sp.diags([self.Aa[1:], self.Ab, self.Ac[:-1]], [-1, 0, 1], format='csc')
        self.Q = sp.eye(len(self.composite.s_grid)-2)
        self.P = self.Q - self.dt * self.A
        self.Pinv = spilu(csc_matrix(self.P), drop_tol=1e-10, fill_factor=100)