import numpy as np
import math
class SDE:
    def MC_matrix(spot,r,q,vol,sim_times,t_day,dt,seed=14):
        #time indicates the expiration day
        M=math.ceil(t_day)
        decimal=t_day-int(t_day)
        a = np.zeros((sim_times, M+1))
        a[:, 0] = spot  # 设置初始股价
        np.random.seed(seed)
        # 生成标准正态分布的随机数矩阵
        Z = np.random.normal(size=(sim_times, M))
        # 计算股价路径
        #首日，考虑小数
        if decimal==0:
            a[:,1]=a[:,0] * np.exp((r-q - 0.5 * vol**2) * (dt) + vol * np.sqrt(dt) * Z[:, 0])
        else:
            a[:,1]=a[:,0] * np.exp((r-q - 0.5 * vol**2) * (dt*decimal) + vol * np.sqrt(dt*decimal) * Z[:, 0])
        #后续整数日不考虑小数
        for t in range(2, M+1):
            a[:, t] = a[:, t-1] * np.exp((r-q - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z[:, t-1])
        return a