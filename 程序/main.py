from options_implement import vanilla_european_options
from options_implement import binary_european_options
from options_implement import barrier_options
import matplotlib.pyplot as plt
import numpy as np

#test binary results
'''
S_values = np.arange(90, 108, 0.2)
option_analy_prices = []
option_mc_prices=[]
for S in S_values:
    print(S)
    price=binary_european_options(option_type='C',spot=S,strike=101,r=0.02,val_date=1,end_date=1/12,vol=5,q=0.02).price()
    option_analy_prices.append(price)
    price=binary_european_options(option_type='C',spot=S,strike=101,r=0.02,val_date=1,end_date=1/12,vol=5,q=0.02).MC_simulation(5000,10)
    option_mc_prices.append(price)
plt.figure(figsize=(10, 6))
# 绘制 analytical solution 的线
plt.plot(S_values, option_analy_prices, label='analytical solutions', linestyle='-', color='blue', linewidth=2)
# 绘制MC method的数值解
plt.plot(S_values, option_analy_prices, label='MC method', linestyle='--', color='black', linewidth=2)
plt.xlabel('asset price $S$')
plt.ylabel('options price $V$')
plt.title('Comparison')
plt.legend()
plt.grid(True)
plt.show()
'''

#test vanilla results
'''
S_values = np.arange(60, 108, 0.2)
option_analy_prices = []
option_mc_prices=[]
for S in S_values:
    print(S)
    price=vanilla_european_options(option_type='C',spot=S,strike=101,r=0.02,val_date=1,end_date=1/12,vol=5,q=0.02).price()
    option_analy_prices.append(price)
    price=vanilla_european_options(option_type='C',spot=S,strike=101,r=0.02,val_date=1,end_date=1/12,vol=5,q=0.02).MC_simulation(5000,10)
    option_mc_prices.append(price)
plt.figure(figsize=(10, 6))
# 绘制 analytical solution 的线
plt.plot(S_values, option_analy_prices, label='analytical solutions', linestyle='-', color='blue', linewidth=2)
# 绘制MC method的数值解
plt.plot(S_values, option_analy_prices, label='MC method', linestyle='--', color='black', linewidth=2)
plt.xlabel('asset price $S$')
plt.ylabel('options price $V$')
plt.title('Comparison')
plt.legend()
plt.grid(True)
plt.show()
'''

#test barrier results
'''
S_values = np.arange(90, 108, 0.2)
option_prices=[]
for S in S_values:
    print(S)
    price = barrier_options(option_type='C',spot=S,strike=100,r=0.02,val_date=1,end_date=1/12,vol=5,B=108).MC_simulation(3000,2000)
    option_prices.append(price)
option_prices_array = np.array(option_prices)
s1,BO1=barrier_options(option_type='C',spot=S_values,strike=100,r=0.02,val_date=1,end_date=1/12,vol=5,B=108,q=0.02).FDM_ununiform(95,108,2000,1000,[100,108],[4,4])
s2,BO2=barrier_options(option_type='C',spot=S_values,strike=100,r=0.02,val_date=1,end_date=1/12,vol=5,B=108,q=0.02).FDM_uniform(95,108,2000,1000)
plt.figure(figsize=(10, 6))
# 绘制 Monte Carlo 方法的线
plt.plot(S_values, option_prices_array, label='MC Method', linestyle='-', color='blue', linewidth=2)
# 绘制不均匀网格的隐式数值解
plt.plot(s1, BO1, label='Implicit Numerical Solution with Ununiform', linestyle='--', color='black', linewidth=2)
# 绘制均匀网格的隐式数值解
plt.plot(s2, BO2, label='Implicit Numerical Solution with Uniform', linestyle=':', color='red', linewidth=2)
plt.xlabel('Asset Price $S$')
plt.ylabel('options Price $V$')
plt.title('Comparison')
plt.legend()
plt.grid(True)
plt.show()
'''