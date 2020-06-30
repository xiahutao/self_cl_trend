# coding=utf-8
'''
Created on 9.30, 2018
适用于btc/usdt，btc计价并结算
@author: fang.zhang
'''
from __future__ import division
from backtest_func import *
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

if __name__ == '__main__':
    os.getcwd()
    print(os.path)
    t0 = time.time()
    symble_lst = ['RB', 'PP', 'TA', 'AG', 'MA', 'JM', 'M']
    # symble_lst = ['M']
    n = 1  # 回测周期
    period = '1d'

    N1_lst = [5]  # 短周期周期1
    N2_lst = [30]  # 长周期周期2
    K1_lst = [i / 100 for i in range(5, 96, 5)]  # 近期比例
    K2_lst = [i / 100 for i in range(5, 96, 5)]  # 长期比例

    P_ATR_LST = [i for i in range(20, 21, 2)]  # ATR周期
    ATR_n_lst = [i / 10 for i in range(20, 21, 2)]  # ATR倍数

    lever_lst = [1]  # 杠杆率，btc初始仓位pos=1,usdt初始仓位pos=0;做多情况下，btc仓位1+lever,usdt仓位-lever;
    # 做空情况下，btc仓位1-lever,usdt仓位lever

    fee = 0.001
    date_lst = [('2010-01-01', '2017-03-01'), ('2017-03-01', '2019-10-01'), ('2010-01-01', '2019-10-01')]
    df_lst = []
    lst = []
    state_lst = []
    for symble in symble_lst:
        group = pd.read_csv('E:/data/' + 'hq_' + period + '_' + symble + '.csv')[
            ['date_time', 'open', 'high', 'low', 'close']]\
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
        group_day = pd.read_csv('E:/data/' + 'hq_' + '1d' + '_' + symble + '.csv')[
            ['date_time', 'open', 'high', 'low', 'close']]\
            .assign(date_time=lambda df: df.date_time.apply(lambda x: str(x)[:10]))
        for N_ATR in P_ATR_LST:
            if len(group_day) < N_ATR:
                continue
            group_day['atr'] = talib.ATR(group_day['high'].values, group_day['low'].values,
                                         group_day['close'].values, N_ATR)

            day_atr = group_day[['date_time', 'atr']] \
                .assign(atr=lambda df: df.atr.shift(1)) \
                .merge(group, on=['date_time'], how='right') \
                .sort_values(['date_time']).fillna(method='ffill')
            print(day_atr)
            for N1 in N1_lst:
                for N2 in N2_lst:
                    if len(day_atr) <= min(N1, N2):
                        continue
                    if N1 < N2:
                        group__ = day_atr \
                            .assign(HH_s=lambda df: talib.MAX(df.high.values, N1)) \
                            .assign(LL_s=lambda df: talib.MIN(df.low.values, N1)) \
                            .assign(HH_l=lambda df: talib.MAX(df.high.values, N2)) \
                            .assign(LL_l=lambda df: talib.MIN(df.low.values, N2)) \
                            .assign(long_ratio=lambda df: ((df.close - df.LL_l) / (df.HH_l - df.LL_l) - 0.5) / 0.5)\
                            .assign(long_ratio=lambda df: df.long_ratio.shift(1))\
                            .assign(short_ratio=lambda df: ((df.close - df.LL_s) / (df.HH_s - df.LL_s) - 0.5) / 0.5)\
                            .assign(short_ratio=lambda df: df.short_ratio.shift(1))\
                            .assign(close_1=lambda df: df.close.shift(1))
                        for (s_date, e_date) in date_lst:
                            group_ = group__[
                                (group__['date_time'] >= s_date) & (group__['date_time'] <= e_date)] \
                                .reset_index(drop=True)
                            print(group_)
                            back_stime = group_.at[0, 'date_time']
                            back_etime = group_.at[len(group_) - 1, 'date_time']

                            for K1 in K1_lst:
                                for K2 in K2_lst:
                                    for ATR_n in ATR_n_lst:
                                        print(symble)
                                        status = -10000
                                        signal_lst = []
                                        trad_times = 0
                                        net = 1
                                        net_lst = []
                                        pos_lst = []
                                        pos = 0
                                        low_price_pre = 0
                                        high_price_pre = 100000000
                                        long_stop = False
                                        short_stop = False
                                        for idx, _row in group_.iterrows():
                                            if pos == 0:
                                                if(_row.long_ratio > K2) & (_row.short_ratio < -K1):
                                                    cost = _row.open * (1 + fee)
                                                    pos = 1
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost/(1+fee))
                                                    high_price.append(cost/(1+fee))
                                                    high_price.append(_row.high)
                                                    net = (pos * _row.close / cost + (1 - pos)) * net

                                                elif(_row.long_ratio < -K2) & (_row.short_ratio > K1):
                                                    cost = _row.open * (1 - fee)
                                                    pos = -1
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    low_price = []
                                                    hold_price.append(cost/(1-fee))
                                                    low_price.append(cost/(1-fee))
                                                    low_price.append(_row.low)
                                                    net = ((1 + pos) - pos * (2 - _row.close / cost)) * net
                                                # elif (_row.high > high_price_pre) & long_stop==True:
                                                #     s_time = _row.date_time
                                                #     cost = high_price_pre * (1 + fee)
                                                #     if _row.open > high_price_pre:
                                                #         cost = _row.open * (1 + fee)
                                                #     pos = 1
                                                #     hold_price = []
                                                #     high_price = []
                                                #     hold_price.append(cost/(1+fee))
                                                #     high_price.append(cost/(1+fee))
                                                #     high_price.append(_row.high)
                                                #     net = (pos * _row.close / cost + (1-pos)) * net
                                                #     long_stop = False
                                                # elif (_row.low < low_price_pre) & short_stop==True:
                                                #     cost = low_price_pre * (1 - fee)
                                                #     if _row.open < low_price_pre:
                                                #         cost = _row.open * (1 - fee)
                                                #     pos = -1
                                                #     s_time = _row.date_time
                                                #     hold_price = []
                                                #     low_price = []
                                                #     hold_price.append(cost/(1-fee))
                                                #     low_price.append(cost/(1-fee))
                                                #     low_price.append(_row.low)
                                                #     net = ((1 + pos) - pos * (2 - _row.close / cost)) * net
                                                #     short_stop = False
                                                else:
                                                    net = net
                                            if pos > 0:
                                                if (_row.long_ratio < -K2) & (_row.short_ratio > K1):
                                                    s_price = _row.open * (1 - fee)
                                                    trad_times += 1
                                                    net1 = pos * s_price / _row.close_1 + (1 - pos)
                                                    ret = s_price / cost - 1
                                                    e_time = _row.date_time
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((max(high_price) / cost) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('spk')
                                                    signal_lst.append(signal_row)
                                                    s_time = _row.date_time
                                                    cost = s_price
                                                    pos = -1
                                                    net2 = (1 + pos) - pos * (2 - _row.close / cost)
                                                    hold_price = []
                                                    low_price = []
                                                    hold_price.append(cost/(1-fee))
                                                    low_price.append(cost/(1-fee))
                                                    low_price.append(_row.low)
                                                    net = net1 * net2 * net
                                                    short_stop = False
                                                    long_stop = False
                                                elif _row.low < max(high_price) - _row.atr * ATR_n:
                                                    trad_times += 1
                                                    e_time = _row.date_time
                                                    s_price = (max(high_price) - _row.atr * ATR_n) * (1 - fee)
                                                    if _row.open < max(high_price) - _row.atr * ATR_n:
                                                        s_price = _row.open * (1 - fee)
                                                    net1 = pos * s_price / _row.close_1 + (1 - pos)
                                                    ret = s_price / cost - 1
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(s_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((max(high_price) / cost) - 1)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('sp')
                                                    signal_lst.append(signal_row)
                                                    pos = 0
                                                    high_price.append(_row.high)
                                                    high_price_pre = max(high_price)
                                                    long_stop = True
                                                    short_stop = False
                                                    net = net1 * net
                                                    status = 0

                                                else:
                                                    high_price.append(_row.high)
                                                    hold_price.append(_row.close)
                                                    net = net * (pos * _row.close / _row.close_1 + (
                                                                    1 - pos))
                                            elif pos < 0:
                                                if(_row.long_ratio > K2) & (_row.short_ratio < -K1):
                                                    b_price = _row.open * (1 + fee)
                                                    e_time = _row.date_time
                                                    trad_times += 1
                                                    net1 = (1 + pos) - pos * (2 - b_price / _row.close_1)
                                                    ret = (cost - b_price) / cost
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((cost - min(low_price)) / cost)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bpk')
                                                    signal_lst.append(signal_row)
                                                    pos = 1
                                                    cost = b_price
                                                    net2 = pos * _row.close / cost + 1 - pos
                                                    s_time = _row.date_time
                                                    hold_price = []
                                                    high_price = []
                                                    hold_price.append(cost/(1+fee))
                                                    high_price.append(cost/(1+fee))
                                                    high_price.append(_row.high)
                                                    net = net1 * net2 * net
                                                    short_stop = False
                                                    long_stop = False

                                                elif _row.high > min(low_price) + _row.atr * ATR_n:
                                                    trad_times += 1
                                                    e_time = _row.date_time
                                                    b_price = (min(low_price) + _row.atr * ATR_n) * (1 + fee)
                                                    if _row.open > min(low_price) + _row.atr * ATR_n:
                                                        b_price = _row.open * (1 + fee)
                                                    net1 = (1 + pos) - pos * (2 - b_price / _row.close_1)
                                                    ret = (cost - b_price) / cost
                                                    signal_row = []
                                                    signal_row.append(s_time)
                                                    signal_row.append(e_time)
                                                    signal_row.append(cost)
                                                    signal_row.append(b_price)
                                                    signal_row.append(ret)
                                                    signal_row.append((cost - min(low_price)) / cost)
                                                    signal_row.append(len(hold_price))
                                                    signal_row.append(pos)
                                                    signal_row.append('bp')
                                                    signal_lst.append(signal_row)
                                                    pos = 0
                                                    low_price.append(_row.low)
                                                    net = net * net1
                                                    low_price_pre = min(low_price)
                                                    short_stop = True
                                                    long_stop = False
                                                    status = 0

                                                else:
                                                    low_price.append(_row.low)
                                                    hold_price.append(_row.close)
                                                    net = net * ((1 + pos) - pos * (2 - _row.close / _row.close_1))
                                            net_lst.append(net)
                                            pos_lst.append(pos)
                                        # net_df = pd.DataFrame({'net': net_lst,
                                        #                        'date_time': group_.date_time.tolist(),
                                        #                        'close': group_.close.tolist(),
                                        #                        'pos': pos_lst}).assign(
                                        #     close=lambda df: df.close / df.close.tolist()[0])
                                        # net_df.to_csv('E:/data/net_ocm.csv')
                                        # net_df[['date_time', 'net', 'close']].plot(
                                        #     x='date_time', kind='line', grid=True,
                                        #     title=symble + '_' + period)
                                        # plt.show()
                                        ann_ROR = annROR(net_lst, n)
                                        total_ret = net_lst[-1]
                                        max_retrace = maxRetrace(net_lst, n)
                                        sharp = yearsharpRatio(net_lst, n)

                                        signal_state = pd.DataFrame(signal_lst, columns=[
                                            's_time', 'e_time', 'b_price', 's_price', 'ret', 'max_ret', 'hold_day',
                                                     'position', 'bspk'])
                                        # signal_state.to_csv('cl_trend/data/signal_gzw_' + str(n) + '_' + symble + '_' + back_stime + 'ls.csv')
                                        # df_lst.append(signal_state)
                                        win_r, odds, ave_r, mid_r = get_winR_odds(signal_state.ret.tolist())
                                        win_R_3, win_R_5, ave_max = get_winR_max(signal_state.max_ret.tolist())
                                        state_row = []
                                        state_row.append(symble)
                                        state_row.append(n)
                                        state_row.append(win_r)
                                        state_row.append(odds)
                                        state_row.append(total_ret - 1)
                                        state_row.append(ann_ROR)
                                        state_row.append(sharp)
                                        state_row.append(max_retrace)
                                        state_row.append(len(signal_state))
                                        state_row.append(ave_r)
                                        state_row.append(signal_state.hold_day.mean())
                                        state_row.append(win_R_3)
                                        state_row.append(win_R_5)
                                        state_row.append(ave_max)

                                        state_row.append(N_ATR)
                                        state_row.append(ATR_n)
                                        state_row.append(N1)
                                        state_row.append(N2)
                                        state_row.append(K1)
                                        state_row.append(K2)
                                        state_row.append(back_stime)
                                        state_row.append(back_etime)
                                        state_lst.append(state_row)

    signal_state = pd.DataFrame(state_lst, columns=['symble', 'tm', 'win_r', 'odds', 'total_ret', 'ann_ret', 'sharp',
                                                    'max_retrace', 'trade_times', 'ave_r', 'ave_hold_days', 'win_r_3',
                                                    'win_r_5', 'ave_max', 'art_N', 'art_n', 'period_s', 'period_l',
                                                    'k_s', 'k_l', 's_time', 'e_time'])
    print(signal_state)
    signal_state.to_csv('E:/data/ocm/' + 'state_ocm_tb_no_add_position' + '_7' + '.csv')
