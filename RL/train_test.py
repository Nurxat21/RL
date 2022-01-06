from RL.Agent.Data import *
from RL.Agent.model import DRLAgent as DRLAgent_erl

def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    # fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    env_config = {'price_array': price_array,
                  'tech_array': tech_array,
                  'turbulence_array': turbulence_array,
                  'if_train': True}
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get('cwd', './' + str(model_name))


    break_step = kwargs.get('break_step', 1e6)
    erl_params = kwargs.get('erl_params')

    agent = DRLAgent_erl(env=env,
                         price_array=price_array,
                         tech_array=tech_array,
                         turbulence_array=turbulence_array)

    model = agent.get_model(model_name, model_kwargs=erl_params)
    trained_model = agent.train_model(model=model,
                                      cwd=cwd,
                                      total_timesteps=break_step)



def test(start_date, end_date, ticker_list, data_source, time_interval,
         technical_indicator_list, drl_lib, env, model_name, if_vix=True,
         **kwargs):
    # fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)

    env_config = {'price_array': price_array,
                  'tech_array': tech_array,
                  'turbulence_array': turbulence_array,
                  'if_train': False}
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get('net_dimension', 2 ** 7)
    cwd = kwargs.get('cwd', './' + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == 'elegantrl':
        episode_total_assets = DRLAgent_erl.DRL_prediction(model_name=model_name,
                                                           cwd=cwd,
                                                           net_dimension=net_dimension,
                                                           environment=env_instance)

        return episode_total_assets
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')