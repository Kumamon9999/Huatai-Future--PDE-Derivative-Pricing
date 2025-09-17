from options_implement import accumulator_forward
import numpy as np
import matplotlib.pyplot as plt
#test MC_method for Accumulator Call

s_values_mc = np.linspace(70,120,251)
option_price_mc=[]
for S in s_values_mc:
    print(S)
    price=accumulator_forward('C',S,95,105,98,90,0.02,0.15,20,0.14,2,10,2,200,0.02).MC_method(5000)
    option_price_mc.append(price)


#test MC_method for Accumulator Put
'''
s_values_mc = np.linspace(70,120,251)
option_price_mc=[]
for S in s_values_mc:
    print(S)
    price=accumulator_forward('P',S,105,95,102,110,0.02,0.15,20,0.14,2,10,2,200,0.02).MC_method(5000)
    option_price_mc.append(price)
'''

#test uniform grid for call


test1=accumulator_forward('C',100,95,105,98,90,0.02,0.15,20,0.14,2,10,2,200,0.02)
s_values_uni=test1.s_uni_grid
s_values_ununi=test1.s_ununi_grid
option_price_uni=test1.FDM_pricing_uniform()
option_price_ununi=test1.FDM_pricing_ununiform()
test1.plt_grid()

'''
test1=accumulator_forward('P',100,105,95,102,110,0.02,0.15,20,0.14,2,10,2,200,0.02)
s_values_uni=test1.s_uni_grid
s_values_ununi=test1.s_ununi_grid
option_price_uni=test1.FDM_pricing_uniform()
option_price_ununi=test1.FDM_pricing_ununiform()
test1.plt_grid()
'''
plt.figure(figsize=(10, 6))
plt.plot(s_values_mc, option_price_mc, label='MC', linestyle=':', color='black', linewidth=0.1, marker='o', markersize=0.5)
plt.plot(s_values_uni, option_price_uni, label='PDE_uni', linestyle='--', color='green', linewidth=0.1, marker='s', markersize=0.5)
plt.plot(s_values_ununi, option_price_ununi, label='PDE_ununi', linestyle=':', color='red', linewidth=0.02, marker='^', markersize=0.1)
#plt.plot(S_values,(option_price_mc-option_price)/option_price,label='Relative Errors', linestyle='-', color='black', linewidth=2)
#plt.plot(S_values,(option_price_mc-option_price),label='Absolute Errors', linestyle='--', color='yellow', linewidth=2)
plt.xlabel('asset price $S$')
plt.ylabel('options price $V$')
plt.title('Comparison')
plt.legend()
plt.grid(True)
plt.show()
