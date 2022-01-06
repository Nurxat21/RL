import datetime
from RL.env import StockTradingEnv
import RL.config as config
from RL.train_test import *
from erl.plot import backtest_stats, backtest_plot










env = StockTradingEnv
TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'
TECHNICAL_INDICATORS_LIST = ['macd',
                             'boll_ub',
                             'boll_lb',
                             'rsi_30',
                             'dx_30',
                             'close_30_sma',
                             'close_60_sma']

ERL_PARAMS = {"learning_rate": 3e-5,"batch_size": 2048,"gamma":  0.985,
              "seed":312,"net_dimension":512, "target_step":5000, "eval_gap":60}

#demo for elegantrl
account_value_train = train(start_date = TRAIN_START_DATE,
                            end_date = TRAIN_END_DATE,
                            ticker_list = config.DOW_30_TICKER,
                            data_source = 'yahoofinance',
                            time_interval= '1D',
                            technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                            drl_lib='elegantrl',
                            env=env,
                            model_name='ddpg',
                            cwd='./test_'+ 'ddpg',
                            erl_params=ERL_PARAMS,
                            break_step=1e5)
account_value_erl=test(start_date = TEST_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = config.DOW_30_TICKER,
                        data_source = 'yahoofinance',
                        time_interval= '1D',
                        technical_indicator_list= TECHNICAL_INDICATORS_LIST,
                        drl_lib='elegantrl',
                        env=env,
                        model_name='ddpg',
                        cwd='./test_ddpg',
                        net_dimension = 512)
####Plot
baseline_df = DataProcessor('yahoofinance').download_data(ticker_list = ["^DJI"],
                                                            start_date = TEST_START_DATE,
                                                            end_date = TEST_END_DATE,
                                                            time_interval = "1D")
stats = backtest_stats(baseline_df, value_col_name = 'close')
account_value_erl = pd.DataFrame({'date':baseline_df.date,'account_value':account_value_erl[0:len(account_value_erl)-1]})
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=account_value_erl)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+"/perf_stats_all_"+".csv.")
print("==============Compare to DJIA===========")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value_erl,
             baseline_ticker = '^DJI',
             baseline_start = account_value_erl.loc[0,'date'],
             baseline_end = account_value_erl.loc[len(account_value_erl)-1,'date'])