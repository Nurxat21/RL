import numpy as np
import torch
import torch.nn as nn


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False), nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)), nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False), nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)), nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    # @staticmethod
    # def check():
    #     inp_dim = 3
    #     out_dim = 32
    #     batch_size = 2
    #     image_size = [224, 112][1]
    #     # from elegantrl.net import Conv2dNet
    #     net = Conv2dNet(inp_dim, out_dim, image_size)
    #
    #     x = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
    #     print(x.shape)
    #     y = net(x)
    #     print(y.shape)

################################################DDPG###################################################################
'''Policy Network (Actor)'''
class Actor(nn.Module):
    """
    A simple Actor class.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :param action_std[float]: the standard deviation of the Gaussian distribution.
        :return: the output tensor.
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

'''Value Network (Critic)'''
class Critic(nn.Module):
    """
    A simple Critic class.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        """
        The forward function.

        :param state[np.array]: the input state.
        :param action[float]: the input action.
        :return: the output tensor.
        """
        return self.net(torch.cat((state, action), dim=1))  # Q value

################################################PPO#####################################################################
class ActorPPO(nn.Module):
    """
    Actor class for **PPO** with stochastic, learnable, **state-independent** log standard deviation.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        """
        The forward function with Gaussian noise.

        :param state[np.array]: the input state.
        :return: the action and added noise.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        """
        Compute the log of probability with current network.

        :param state[np.array]: the input state.
        :param action[float]: the action.
        :return: the log of probability and entropy.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        """
        Compute the log of probability with old network.

        :param _action[float]: the action.
        :param noise[float]: the added noise when exploring.
        :return: the log of probability with old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

class ActorDiscretePPO(nn.Module):
    """
    Actor class for **Discrete PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        """
        The forward function with Softmax.

        :param state[np.array]: the input state.
        :return: the action index and probabilities.
        """
        a_prob = self.soft_max(self.net(state))
        # dist = Categorical(a_prob)
        # a_int = dist.sample()
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
        # samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        # samples_2d.shape == (batch_size, num_samples)
        return a_int, a_prob

    def get_logprob_entropy(self, state, a_int):
        """
        Compute the log of probability with current network.

        :param state[np.array]: the input state.
        :param a_int[int]: the action.
        :return: the log of probability and entropy.
        """
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        """
        Compute the log of probability with old network.

        :param a_int[int]: the action.
        :param a_prob[float]: the action probability.
        :return: the log of probability with old network.
        """
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)

class CriticPPO(nn.Module):
    """
    The Critic class for **PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=0.5)  # output layer for advantage value

    def forward(self, state):
        """
        The forward function to ouput the value of the state.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # advantage value

class SharePPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        if isinstance(state_dim, int):
            self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim))  # the only difference.
        else:
            self.enc_s = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])
        out_dim = mid_dim

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))
        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)

        self.dec_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))
        self.dec_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))

        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get_action_noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get_q_logprob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        logprob = -(noise.pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q, logprob

    def get_q1_q2_logprob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()
        logprob = -(((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q1, q2, logprob

#################################################TD3####################################################################
class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)

class CriticTwin(nn.Module):  # shared parameter
    """
    The Critic class for **Clipped Double DQN**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, inp_dim), nn.ReLU(),
                                    nn_middle, )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        """
        The forward function to ouput a single Q-value.

        :param state[np.array]: the input state.
        :param action[float]: the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        """
        The forward function to output two Q-values from two shared-paramter networks.

        :param state[np.array]: the input state.
        :param action[float]: the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

##################################################SAC##################################################################
class ActorSAC(nn.Module):
    """
    Actor class for **SAC** with stochastic, learnable, **state-dependent** log standard deviation..

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                       nn_middle, )
        self.net_a_avg = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state):
        """
        The forward function.

        :param state[np.array]: the input state.
        :return: the output tensor.
        """
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        """
        The forward function with noise.

        :param state[np.array]: the input state.
        :return: the action and added noise.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        """
        Compute the action and log of probability with current network.

        :param state[np.array]: the input state.
        :return: the action and log of probability.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        '''add noise to a_noise in stochastic policy'''
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of a_noise (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        """same as below:
        from torch.distributions.normal import Normal
        log_prob = Normal(a_avg, a_std).log_prob(a_noise)
        # same as below:
        a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        log_prob = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        """

        '''fix log_prob of action.tanh'''
        log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        """same as below:
        epsilon = 1e-6
        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob - (1 - a_noise_tanh.pow(2) + epsilon).log()

        Thanks for:
        https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/actor.py#L37
        ↑ MIT License， Thanks for https://www.zhihu.com/people/Z_WXCY 2ez4U
        They use action formula that is more numerically stable, see details in the following link
        https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#TanhTransform
        https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
        """
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)

########################################################################################################################
