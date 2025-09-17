# for other models, only need to change the SDE to adapt to new MC methods.
import numpy as np
class SDE:
    def MC_matrix(self,spot_0,r,q,vol,T,sim_times=5000,t_steps=5000):
        dt=T/sim_times
        #N indicates the MC times
        #M indicates time intervals
        # 初始化股价矩阵
        a = np.zeros((sim_times, t_steps+1))
        a[:, 0] = spot_0  # 设置初始股价
        # 生成标准正态分布的随机数矩阵
        Z = np.random.normal(size=(sim_times, t_steps))
        # 计算股价路径
        for t in range(1, t_steps+1):
            a[:, t] = a[:, t-1] * np.exp((r-q) * dt) + vol * np.sqrt(np.exp(2*(r-q)*dt)-1)/np.sqrt(2*(r-q)) * Z[:, t-1]
        return a