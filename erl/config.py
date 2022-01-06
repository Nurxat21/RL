import os
import torch
import numpy as np


class Arguments:  # [ElegantRL.2021.10.21]
    """
        Configuration map.

        :param env[object]: the environment object in ElegantRL.
        :param agent[object]: the agent object in ElegantRL.

        **Attributes for environment setup**

        Attributes
        ----------------
            env : object
                environment object in ElegantRL.
            env_num : int
                number of sub-environments. For VecEnv, env_num > 1.
            max_step : int
                max step of an episode.
            state_dim : int
                state dimension of the environment.
            action_dim : int
                action dimension of the environment.
            if_discrete : boolean
                discrete or continuous action space.
            target_return : float
                target average episodic return.

        **Attributes for model training**

        Attributes
        ----------------
            agent : object
                agent object in ElegantRL.
            if_off_policy : boolean
                off-policy or on-policy for the DRL algorithm.
            net_dim : int
                neural network width.
            max_memo : int
                capacity of replay buffer.
            batch_size : int
                number of transitions sampled in one iteration.
            target_step : int
                repeatedly update network to keep critic's loss small.
            repeat_times : int
                collect target_step, then update network.
            break_step : int
                break training after total_step > break_step.
            if_allow_break : boolean
                allow break training when reach goal (early termination).
            if_per_or_gae : boolean
                use Prioritized Experience Replay (PER) or not for off-policy algorithms.

                use Generalized Advantage Estimation or not for on-policy algorithms.
            gamma : float
                discount factor of future rewards.
            reward_scale : int
                an approximate target reward.
            learning_rate : float
                the learning rate.
            soft_update_tau : float
                soft update parameter for target networks.

        **Attributes for model evaluation**

        Attributes
        ----------------
            eval_env : object
                environment object for model evaluation.
            eval_gap : int
                time gap for periodical evaluation (in seconds).
            eval_times1 : int
                number of times that get episode return in first.
            eval_times2 : int
                number of times that get episode return in second.
            eval_gpu_id : int or None
                the GPU id for the evaluation environment.

                -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0].
            if_overwrite : boolean
                save policy networks with different episodic return separately or overwrite.

        **Attributes for resource allocation**

        Attributes
        ----------------
            worker_num : int
                rollout workers number per GPU (adjust it to get high GPU usage).
            thread_num : int
                cpu_num for evaluate model.
            random_seed : int
                initialize random seed in ``init_before_training``.
            learner_gpus : list
                GPU ids for learner.
            workers_gpus : list
                GPU ids for worker.
            ensemble_gpus : list
                GPU ids for population-based training (PBT).
            ensemble_gap : list
                time gap for leaderboard update in tournament-based ensemble training.
            cwd : string
                directory path to save the model.
            if_remove : boolean
                remove the cwd folder? (True, False, None:ask me).
    """
    def __init__(self, env, agent):
        self.env = env  # the environment for training
        self.env_num = getattr(env, 'env_num', 1)  # env_num = 1. In vector env, env_num > 1.

        self.max_step = getattr(env, 'max_step', None)  # the max step of an episode
        self.state_dim = getattr(env, 'state_dim', None)  # vector dimension (feature number) of state
        self.action_dim = getattr(env, 'action_dim', None)  # vector dimension (feature number) of action
        self.if_discrete = getattr(env, 'if_discrete', None)  # discrete or continuous action space
        self.target_return = getattr(env, 'target_return', None)  # target average episode return

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.if_off_policy = agent.if_off_policy  # agent is on-policy or off-policy
        if self.if_off_policy:  # off-policy
            self.net_dim = 2 ** 8  # the network width
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER (Prioritized Experience Replay) for sparse reward
        else:  # on-policy
            self.net_dim = 2 ** 9  # the network width
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = (0,)  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.workers_gpus = self.learner_gpus  # for GPU_VectorEnv (such as isaac gym)
        self.ensemble_gpus = None  # for example: (learner_gpus0, ...)
        self.ensemble_gap = 2 ** 8

        self.cwd = None  # the directory path to save the model
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 8  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_gpu_id = None  # -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0]
        self.if_overwrite = True  # Save policy networks with different episode return or overwrite

    def init_before_training(self):
        """
            Check parameters before training.
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''env'''
        assert isinstance(self.env_num, int)
        assert isinstance(self.max_step, int)
        assert isinstance(self.state_dim, int) or isinstance(self.state_dim, tuple)
        assert isinstance(self.action_dim, int) or isinstance(self.action_dim, tuple)
        assert isinstance(self.if_discrete, bool)
        assert isinstance(self.target_return, int) or isinstance(self.target_return, float)

        '''agent'''
        assert hasattr(self.agent, 'init')
        assert hasattr(self.agent, 'update_net')
        assert hasattr(self.agent, 'explore_env')
        assert hasattr(self.agent, 'select_actions')

        '''auto set'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            self.cwd = f'./{agent_name}_{env_name}_{self.learner_gpus}'
        if self.eval_gpu_id is None:
            self.eval_gpu_id = self.learner_gpus[0]

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Remove cwd: {self.cwd}")
        else:
            print(f"| Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)