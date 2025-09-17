from snowball  import *
import matplotlib.pyplot as plt

class Standard_Snowball(Snowball):

    def __init__(
            self,
            val_date,                               # 估值日期
            end_date,                               # 到期日
            notional,                               #nominal capital
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
            ki_flag=False,                          # whether knock_in
            year_base=245,                          # 每年的天数， 交易日算一般定为245
            s_steps=2000,                           # S网格点切分的个数
            concentrate=True
    ):
        super().__init__(
            notional=notional,
            s0=s0,
            val_date=val_date,
            end_date=end_date,
            spot=spot,
            vol=vol,
            r=r,
            q=q,
            ki_flag=ki_flag,
            ko_coupon=ko_coupon,
            no_ko_coupon=no_ko_coupon,
            ko_barrier=ko_barrier,
            #ko_obs_dates=ko_obs_dates,
            strike=strike,
            option_type=option_type,
            year_base=year_base,
            s_steps=s_steps,
            time_steps_per_day=1,
            concentrate=concentrate
        )

    def daily_payment(self, spot):
        if self.sign * (spot - self.strike)>=0 and  self.sign * (spot -self.ko_barrier)<0:
            payment=self.no_ko_coupon/self.year_base*self.s0
        else:
            payment =0
        return payment



if __name__ == "__main__":
    price_con=[]
    price_uni=[]
    s_value=np.linspace(80,160,101)
    for s in s_value:
        print(s,":")
        paras = {
            'notional':10000,
            's0':100,
            'val_date':0,
            'end_date': 245,
            'spot': s,
            'vol': 0.22,
            'r': 0.03,
            'q': 0.03,
            'strike': 100,
            'ko_barrier': 150,
            's_steps': 2000,
            'option_type': 'call',
            'concentrate':True,
            'ki_flag':True,
            'ko_coupon':0.3,
            'no_ko_coupon':0.2,
        }
        model = Standard_Snowball(**paras)
        price_con.append(model.get_pv())
        print(model.get_pv())
    for s in s_value:
        print(s,":")
        paras = {
            'notional':10000,
            's0':100,
            'val_date':0,
            'end_date': 245,
            'spot': s,
            'vol': 0.22,
            'r': 0.03,
            'q': 0.03,
            'strike': 100,
            'ko_barrier': 150,
            's_steps': 2000,
            'option_type': 'call',
            'concentrate':False,
            'ki_flag':False,
            'ko_coupon':0.3,
            'no_ko_coupon':0.2,
        }
        model = Standard_Snowball(**paras)
        price_uni.append(model.get_pv())
        print(model.get_pv())
    plt.figure(figsize=(10, 6))
    plt.plot(s_value,price_con, label='Concentrate', linestyle=':', color='black', linewidth=0.1, marker='o', markersize=0.5)
    plt.plot(s_value,price_uni, label='Uniform', linestyle=':', color='blue', linewidth=0.1, marker='o', markersize=0.5)
    plt.xlabel('asset price $S$')
    plt.ylabel('options price $V$')
    plt.title('Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

