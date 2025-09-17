'''import numpy as np
import matplotlib.pyplot as plt
from Options_Implement import Vanilla_European_Options
class FDM_Solver:
    def __init__(self, S_max, S_min, T, Time_Steps, Asset_Price_Steps, K, r, q, v, type):
        self.S_max = S_max
        self.S_min = S_min
        self.T = T
        self.Time_Steps = Time_Steps  # Number of time steps
        self.Asset_Price_Steps = Asset_Price_Steps  # Number of asset price steps
        self.K = K  # Strike price
        self.r = r  # Risk-free interest rate
        self.q = q  # Dividend yield
        self.v = v  # Volatility
        self.type = type  # Option type: C or P

        self.dt = T / Time_Steps
        self.dS = (S_max - S_min) / Asset_Price_Steps
        self.S = np.linspace(S_min, S_max, Asset_Price_Steps + 1)
        self.t = np.linspace(0, T, Time_Steps + 1)
        self.V = np.zeros(Asset_Price_Steps + 1)  # Initialize V as a NumPy array

    def setup_initial_conditions(self):
        # Set up initial condition (option payoff at maturity)
        if self.type == 'C':  # Call option
            self.V = np.maximum(self.S - self.K, 0)
        elif self.type == 'P':  # Put option
            self.V = np.maximum(self.K - self.S, 0)
        else:
            raise ValueError("Invalid option type. Use P for put and C for call.")

    def setup_boundary_conditions(self):
        # Set up boundary conditions
        self.V[0] = 0  # V(0, t) = 0
        self.V[-1] = Vanilla_European_Options('C',self.H,self.K,self.T,self.r,self.v,self.q).Price()   # V(H, t) = c(S,K,t,r,T,q)

    def explicit_fd(self):
        # Explicit finite difference method
        self.setup_initial_conditions()

        # Time stepping
        for m in range(self.Time_Steps - 1, -1, -1):
            for n in range(1, self.Asset_Price_Steps):
                S = self.S[n]
                a = 0.5 * (self.v**2* self.dt / self.dS**2 + (self.r - self.q) * S * self.dt / self.dS)
                b = 1 - self.v**2 * self.dt / self.dS**2 - self.r * self.dt
                c = 0.5 * (self.v**2 * self.dt / self.dS**2 - (self.r - self.q) * S * self.dt / self.dS)
                self.V[n] = a * self.V[n + 1] + b * self.V[n] + c * self.V[n - 1]
                
            # Apply boundary conditions at each time step
            self.V[0] = 0
            self.V[-1] = Vanilla_European_Options('C',self.H,self.K,self.T,self.r,self.v,self.q).Price()   # V(H, t) = c(S,K,t,r,T,q)

    def plot_solution(self):
        # Plot the solution
        plt.plot(self.S, self.V, label=f't = {self.T}')
        plt.xlabel('Underlying Asset Price (S)')
        plt.ylabel('Option Value (V)')
        plt.title('Black-Scholes PDE Solution using Explicit FDM')
        plt.legend()
        plt.show()
'''