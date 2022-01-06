import torch
from RL.Agent.Algo.Need.AgentPPO import AgentPPO

class AgentA2C(AgentPPO):  # A2C.2015, PPO.2016
    """
    Bases: ``AgentPPO``

    A2C algorithm. “Asynchronous Methods for Deep Reinforcement Learning”. Mnih V. et al.. 2016.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentPPO.__init__(self)
        print('| AgentA2C: A2C or A3C is worse than PPO. We provide AgentA2C code just for teaching.'
              '| Without TrustRegion, A2C needs special hyper-parameters, such as smaller repeat_times.')

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            # buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: advantage_value in ReplayBuffer
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            # logprob = buf_logprob[indices]

            '''A2C: Advantage function'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            obj_actor = -(adv_v * new_logprob.exp()).mean() + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple