import math
import numpy as np
from scipy.stats import norm
from options_module import options
from options_module import european_options
from SDE_class import SDE
from grid import CompositeTransformation
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

class vanilla_european_options(european_options):
    def __init__(self, option_type, spot, strike, r, val_date,end_date, vol,q=0):
        super().__init__(option_type, spot, strike, r, val_date,end_date,vol,q=0)

    def d(self):
        if self.r==self.q:
            variance=self.vol*math.sqrt(self.T)
        else:
            variance=math.sqrt(self.vol*self.vol*(math.exp(2*(self.r-self.q)*self.T)-1)/(2*(self.r-self.q)))
        if(variance<=0):              #in case variance=0
            variance=0.0001
        return (self.spot*math.exp((self.r-self.q)*(self.T))-self.strike)/(variance)
    
    def price(self):
        value=0
        if self.r==self.q:
            variance=self.vol*math.sqrt(self.T)
        else:
            variance=math.sqrt(self.vol*self.vol*(math.exp(2*(self.r-self.q)*self.T)-1)/(2*(self.r-self.q)))
        if(self.option_type=='C'):
            value=math.exp(-self.r*(self.T))*(self.spot*math.exp((self.r-self.q)*(self.T))*norm.cdf(self.d())
                            +variance*norm.pdf(self.d()))-self.strike*math.exp(-self.r*self.T)*norm.cdf(self.d())
        if(self.option_type=='P'):
            value=math.exp(-self.r*(self.T))*(-self.spot*math.exp((self.r-self.q)*(self.T))*norm.cdf(-self.d())
                            +variance*norm.pdf(self.d()))+self.strike*math.exp(-self.r*self.T)*norm.cdf(-self.d())
        return value
    
    def future_payoff(self,S_T):
        if(self.option_type=='C'):
            return max(0,S_T-self.strike)
        if(self.option_type=='P'):
            return max(self.strike-S_T,0)
    
    def delta(self,h=1e-2)->float:
        positive=vanilla_european_options(self.option_type, self.spot*(1+h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        negative=vanilla_european_options(self.option_type, self.spot*(1-h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        return (positive.price()-negative.price())/(2*self.spot*h)
    
    def gamma(self,h=1e-2)->float:
        positive=vanilla_european_options(self.option_type, self.spot*(1+h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        negative=vanilla_european_options(self.option_type, self.spot*(1-h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        return (positive.price()+negative.price()-2*self.price())/((self.spot*h)*(self.spot*h))
    
    def vega(self,h=1e-2)->float:
        positive=vanilla_european_options(self.option_type, self.spot, self.strike, self.r,self.val_date,self.end_date, self.vol+h)
        return positive.price()-self.price()
    
    def theta(self,h=1)->float:
        positive=vanilla_european_options(self.option_type, self.spot, self.strike, self.r,self.val_date,self.end_date+h, self.vol)
        return positive.price()-self.price()
    
    def rho(self,h=1e-4)->float:
        positive=vanilla_european_options(self.option_type, self.spot, self.strike, self.r+h,self.val_date,self.end_date, self.vol)
        return positive.price()-self.price()

    def MC_simulation(self,sim_times=5000,t_steps=5000):
        V=SDE().MC_matrix(self.spot,self.r,self.q,self.vol,self.T,sim_times,t_steps)
        price=0
        for i in range(0, sim_times):
            tmp=self.future_payoff(V[i,t_steps])
            price += tmp /sim_times
        price *= np.exp(-self.r * self.T)
        return price 

class binary_european_options(european_options):
    def __init__(self, option_type, spot, strike, r, val_date,end_date, vol,q=0):
        super().__init__(option_type, spot, strike, r, val_date,end_date,vol,q=0)
        
    def d(self):
        if self.r==self.q:
            variance=self.vol*math.sqrt(self.T)
        else:
            variance=math.sqrt(self.vol*self.vol*(math.exp(2*(self.r-self.q)*self.T)-1)/(2*(self.r-self.q)))
        if(variance<=0):              #in case variance=0
            variance=0.0001
        return (self.spot*math.exp((self.r-self.q)*(self.T))-self.strike)/(variance)
    
    def price(self)->float:
        if(self.option_type=='C'):
            return math.exp(-self.r*self.T)*norm.cdf(self.d())
        if(self.option_type=='P'):
            return math.exp(-self.r*self.T)*norm.cdf(-self.d())
        
    def future_payoff(self,S_T)->float:
        if(self.option_type=='C'):
            return 1.0 if S_T-self.strike>=0 else 0.0
        if(self.option_type=='P'):
            return 1.0 if S_T-self.strike<=0 else 0.0

    def delta(self,h=1e-2)->float:
        positive=binary_european_options(self.option_type, self.spot*(1+h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        negative=binary_european_options(self.option_type, self.spot*(1-h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        return (positive.price()-negative.price())/(2*self.spot*h)
    
    def gamma(self,h=1e-2)->float:
        positive=binary_european_options(self.option_type, self.spot*(1+h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        negative=binary_european_options(self.option_type, self.spot*(1-h), self.strike, self.r,self.val_date,self.end_date, self.vol)
        return (positive.price()+negative.price()-2*self.price())/((self.spot*h)*(self.spot*h))
    
    def vega(self,h=1e-2)->float:
        positive=binary_european_options(self.option_type, self.spot, self.strike, self.r,self.val_date,self.end_date, self.vol+h)
        return positive.price()-self.price()
    
    def theta(self,h=1)->float:
        positive=binary_european_options(self.option_type, self.spot,self.strike, self.r,self.val_date,self.end_date+h, self.vol)
        return positive.price()-self.price()
    
    def rho(self,h=1e-4)->float:
        positive=binary_european_options(self.option_type, self.spot, self.strike, self.r+h,self.val_date,self.end_date, self.vol)
        return positive.price()-self.price()

    def MC_simulation(self,sim_times=5000,t_steps=5000):
        V=SDE().MC_matrix(self.spot,self.r,self.q,self.vol,self.T,sim_times,t_steps)
        price=0
        for i in range(0, sim_times):
            tmp=self.future_payoff(V[i,t_steps])
            price += tmp /sim_times
        price *= np.exp(-self.r * self.T)
        return price 
    
class barrier_options(european_options):
    def __init__(self, option_type, spot, strike, r, val_date,end_date, vol,B,q=0):
        super().__init__(option_type, spot, strike, r, val_date,end_date,vol,q=0)
        self.B=B
    
    def future_payoff(self,S_T,Touch_Barrier)->float:
        if(Touch_Barrier==False): return max(S_T-self.strike,0);
        return 0
    
    def MC_simulation(self,sim_times=5000,t_steps=5000):
        V=SDE().MC_matrix(self.spot,self.r,self.q,self.vol,self.T,sim_times,t_steps)
        price=0
        for i in range(0, sim_times):
            Touch_Barrier=False
            for t in range(1,t_steps):
                if(V[i,t]>=self.B):
                    Touch_Barrier=True
                    break
            tmp=self.future_payoff(V[i,t_steps],Touch_Barrier)
            price += tmp /sim_times
        price *= np.exp(-self.r * self.T)
        return price
        
    def FDM_ununiform(self,s_min,s_max,s_steps,t_steps,x_star_all, a_all,):
        composite = CompositeTransformation(x_star_all, a_all,s_min,s_max,s_steps)
        u_grid = np.linspace(s_min,s_max, s_steps+1)
        du=(s_max-s_min)/s_steps
        dt=self.T/t_steps
        A=composite.compute_A()
        s_grid= composite.compute_grid(A)
        second_orders = composite.compute_derivative()
        first_orders= composite.__call__(u_grid,s_grid,A)
        a=self.vol*self.vol/(2*first_orders*first_orders)
        b=(-second_orders*self.vol*self.vol)/(2*np.power(first_orders,3))+self.r*s_grid/first_orders
        #print(f'x_grid: {s_grid}')  # 打印 s_grid 以确保其内容正确
        Aa = a[1:s_steps]/(du**2)-b[1:s_steps]/(2*du)
        Ab = -2*a[1:s_steps]/(du**2)-self.r
        Ac = a[1:s_steps]/(du**2)+b[1:s_steps]/(2*du)
        A = sp.diags([Aa[1:], Ab, Ac[:-1]], [-1, 0, 1], format='csc')
        Q = sp.eye(len(s_grid)-2)
        P = Q - dt * A
        Pinv = spilu(csc_matrix(P), drop_tol=1e-10, fill_factor=100)
        pv_matrix=np.zeros((s_steps+1,t_steps+1))
        for i in range (s_steps+1):
            pv_matrix[i,t_steps]=np.maximum(s_grid[i]-self.strike,0)
        for i in range(t_steps-1, -1, -1):
            Z = pv_matrix[1: -1, i + 1].copy()
            Z[0] += Aa[0] * dt * pv_matrix[0, i]
            Z[-1] += Ac[-1] * dt * pv_matrix[-1, i]
            pv_matrix[1: -1, i] = Pinv.solve(Z)
        # 绘制比较图
        '''plt.figure(figsize=(10, 6))
        plt.plot(s_grid, pv_matrix[:,0], label='Implicit Numerical Solution', linestyle='--')
        plt.xlabel('Asset price $S$')
        plt.ylabel('options price $V$')
        plt.title('Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()'''
        return s_grid,pv_matrix[:,0]

    def FDM_uniform(self,s_min,s_max,s_steps,t_steps):
        s_grid = np.linspace(s_min,s_max, s_steps+1)
        du=(s_max-s_min)/s_steps
        dt=self.T/t_steps
        Aa = 0.5 * ((self.vol/du) ** 2 - (self.r)*s_grid[1:-1]/du)
        Ab = -(self.vol/du) ** 2 - self.r
        Ac = 0.5 * ((self.vol/du) ** 2 + (self.r)*s_grid[1:-1]/du)
        A = sp.diags([Aa[1:], Ab, Ac[:-1]], [-1, 0, 1], format='csc')
        Q = sp.eye(len(s_grid)-2)
        P = Q - dt * A
        Pinv = spilu(csc_matrix(P), drop_tol=1e-10, fill_factor=100)
        pv_matrix=np.zeros((s_steps+1,t_steps+1))
        for i in range (s_steps+1):
            pv_matrix[i,t_steps]=np.maximum(s_grid[i]-self.strike,0)
        for i in range(t_steps-1, -1, -1):
            Z = pv_matrix[1: -1, i + 1].copy()
            Z[0] += Aa[0] * dt * pv_matrix[0, i]
            Z[-1] += Ac[-1] * dt * pv_matrix[-1, i]
            pv_matrix[1: -1, i] = Pinv.solve(Z)
        # 绘制比较图
        '''plt.figure(figsize=(10, 6))
        plt.plot(s_grid, pv_matrix[:,0], label='Implicit Numerical Solution', linestyle='--')
        plt.xlabel('Asset price $S$')
        plt.ylabel('options price $V$')
        plt.title('Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()'''
        return s_grid,pv_matrix[:,0]