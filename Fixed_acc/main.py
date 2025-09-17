from fixed_acc_barrier_ki import *
import matplotlib.pyplot as plt

class FixedAccBarrierKI(BasePricerAcc):

    """
    建仓宝1.0（敲入后立即转为线性）
    固赔建仓累计（带敲出）
    - binary with barrier
    """

    def __init__(
            self,
            val_date,                       # 估值日期
            end_date,                       # 到期日
            spot,                           # 现价
            vol,                            # 波动率
            r,                              # 无风险利率
            q,                              # 红利
            strike,                         # 执行价
            barrier,                        # 障碍价
            ki_barrier,                     # 敲入障碍价
            ki_strike,                      # 建仓价
            payoff,                         # 赔付
            leverage,                       # 每日线性倍数
            final_position_quantity,        # 最后一日的线性倍数
            base_quantity,                  # 每日数量
            option_type,                    # 期权类型
            obs_price_list=None,            # 已过观察日的收盘价格序列，默认为None，不输入则不考虑已累计部分的估值
            year_base=245,                  # 年化天数，交易日，一般定为245
            s_steps=2000,                   # S网格点切分的个数
            concentrate=True
    ):
        self.barrier = barrier
        self.payoff = payoff
        super().__init__(
            val_date=val_date,
            end_date=end_date,
            spot=spot,
            vol=vol,
            r=r,
            q=q,
            barrier=barrier,
            strike=strike,
            ki_barrier=ki_barrier,
            ki_strike=ki_strike,
            leverage=leverage,
            final_position_quantity=final_position_quantity,
            base_quantity=base_quantity,
            option_type=option_type,
            obs_price_list=obs_price_list,
            year_base=year_base,
            s_steps=s_steps,
            time_steps_per_day=1,
            concentrate=concentrate
        )

    def daily_payment(self, spot):
        if self.sign * (spot - self.ki_barrier) >= 0:
            if (self.sign * (spot - self.barrier) < 0) & (self.sign * (spot - self.strike) >= 0):
                payment = self.payoff
            else:
                payment = 0
        else:
            payment = self.sign * self.leverage * (spot - self.ki_strike)
        return self.base_quantity * payment


if __name__ == "__main__":
    price_con=[]
    price_uni=[]
    s_value=np.linspace(99.95,100.05,51)
    for s in s_value:
        print(s,":")
        paras = {
            'val_date':9.9999,
            'end_date': 10,
            'spot': s,
            'vol': 0.22,
            'r': 0.03,
            'q': 0.03,
            'strike': 100,
            'barrier': 110,
            'payoff': 10,
            'base_quantity': 1,
            'ki_barrier': 95,
            'ki_strike': 95,
            'leverage': 2,
            'final_position_quantity': 10,
            's_steps': 2000,
            'option_type': 'call',
            'concentrate':True
        }
        model = FixedAccBarrierKI(**paras)
        price_con.append(model.get_pv())
        print(model.get_pv())
    for s in s_value:
        print(s,":")
        paras = {
            'val_date':9.99,
            'end_date': 10,
            'spot': s,
            'vol': 0.22,
            'r': 0.03,
            'q': 0.03,
            'strike': 100,
            'barrier': 110,
            'payoff': 10,
            'base_quantity': 1,
            'ki_barrier': 95,
            'ki_strike': 95,
            'leverage': 2,
            'final_position_quantity': 10,
            's_steps': 2000,
            'option_type': 'call',
            'concentrate':False
        }
        model = FixedAccBarrierKI(**paras)
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

