from abc import abstractmethod,ABC
import math
from scipy.stats import norm

class options(ABC):
    def __init__(self, option_type, spot, strike, r, val_date,end_date, vol,q=0):
        if option_type not in ('C', 'P'):
            raise ValueError("category must be 'C' or 'P'")
        self.T = end_date-val_date+1              # expire time, day!!
        self.strike = strike              # striking price
        self.vol = vol              # volatility
        self.r = r              # non-risk rate
        self.spot = spot              # stock price
        self.option_type = option_type        # options type：'C'or'P'
        self.val_date=val_date
        self.end_date=end_date
        self.q=q

class european_options(options):      #a derived class in Options class
    def __init__(self, option_type, spot, strike, r, val_date,end_date, vol,q=0):
        super().__init__(option_type, spot, strike, r, val_date,end_date,vol,q=0)
