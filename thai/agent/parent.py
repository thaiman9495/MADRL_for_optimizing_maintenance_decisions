import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import StepLR
from collections import deque
from thai.env.system import System
from thai.buffer.uniform_buffer import UniformBuffer
from thai.utility.action_mask import create_action_mask, create_action_mask_single_agent
from thai.utility.state_action_space import create_state_to_id, create_action_space, create_action_to_id


class Agent:
    def __init__(self, n_components, n_c_actions, n_c_states,  device, is_single_agent, **param):
        # parameters of interactive system
        self.n_components = n_components
        self.n_c_actions = n_c_actions
        self.n_c_states = n_c_states
        self.is_stochastic_dependence = param['is_stochastic_dependence']

        # agent characteristics
        self.is_single_agent = is_single_agent

        # create action space
        if self.is_single_agent:
            self.action_space = create_action_space(self.n_components, self.n_c_actions)
            self.n_s_actions = len(self.action_space)
            self.action_to_id = create_action_to_id(self.n_components, self.n_c_actions)

        # parameters for training agents
        self.device = device
        self.epsilon_start = param['epsilon_start']
        self.epsilon_end = param['epsilon_end']
        self.epsilon_anneal_time = param['epsilon_anneal_time']
        self.gamma = param['discount_factor']
        self.target_update_freq = param['target_update_freq']
        self.policy_update_freq = param['policy_update_freq']
        self.log_freq = param['log_freq']
        self.batch_size = param['batch_size']
        self.buffer_capacity = param['buffer_capacity']
        self.n_training_steps = param['n_steps_training']
        self.lr_end = param['lr_end']
        self.grad_norm_clip = param['grad_norm_clip']
        self.reward_normalization = param['reward_normalization']

        # Evaluating trained agents
        self.n_runs = param['n_runs']
        self.episode_len_eval = param['episode_len_eval']
        self.episode_len_train = param['episode_len_train']

        # Action masking
        if self.is_single_agent:
            p_temp = self.n_components, self.n_c_states, self.n_c_actions
            self.action_mask = create_action_mask_single_agent(*p_temp).to(device)
            self.state_to_id = create_state_to_id(self.n_components, self.n_c_states)

        else:
            self.action_mask = create_action_mask(self.n_c_states, self.n_c_actions).to(device)

    @abstractmethod
    def update_target_net(self):
        pass

    @abstractmethod
    def update_policy(self, step):
        pass

    @abstractmethod
    def train(self, env, path_model, path_log):
        pass

    @abstractmethod
    def evaluate(self, env, path_model, path_log):
        pass

    @staticmethod
    def save_policy(q_net, step, path: Path):
        torch.save(q_net.state_dict(), path.joinpath(f'policy_{step}.pt'))

    @staticmethod
    def exploration_shudeler(step, epsilon_start: float, epsilon_end: float, epsilon_anneal_time: int):
        if step <= epsilon_anneal_time:
            return step * (epsilon_end - epsilon_start) / epsilon_anneal_time + epsilon_start
        return epsilon_end

    def choose_action(self, q_net, state, epsilon: float):
        a = np.random.uniform()

        # --------------------------------------------------------------------------------------------------------------
        # Choose action randomly
        # --------------------------------------------------------------------------------------------------------------
        if a < epsilon:
            joint_action = []
            for idx, s_ in enumerate(state):
                s = int(s_)
                if s == 0:
                    a = 0
                else:
                    if s == 3:
                        a = np.random.choice([0, 2])
                    else:
                        a = np.random.randint(self.n_c_actions)
                joint_action.append(a)

            if self.is_single_agent:
                return self.action_to_id[tuple(joint_action)]
            else:
                return np.array(joint_action, dtype=int)

        # --------------------------------------------------------------------------------------------------------------
        # Choose action according to learned policies
        # --------------------------------------------------------------------------------------------------------------
        state_t = torch.atleast_2d(torch.FloatTensor(state).to(self.device))

        # Single agent
        if self.is_single_agent:
            q = torch.squeeze(q_net(state_t), dim=0).detach()
            state_id = self.state_to_id[tuple(state)]
            q += self.action_mask[state_id]
            return torch.argmax(q).cpu().numpy()

        # Multi-agent
        q = torch.squeeze(q_net(state_t), dim=1).detach()
        mask = torch.stack([self.action_mask[int(local_s)] for local_s in state], dim=0)
        q += mask
        return torch.argmax(q, dim=1, keepdim=False).cpu().numpy()

    def train_(self, q_net, buffer: UniformBuffer, lr_scheduler: StepLR, env: System, path_model: Path, path_log: Path):
        # Prepare
        cost_holder = deque(maxlen=5000)
        list_cost_rate = []
        list_step = []

        # Main training loop
        starting_time = datetime.datetime.now()
        for step in range(self.n_training_steps):
            if step % self.episode_len_train == 0:
                env.reset()

            epsilon = self.exploration_shudeler(step, self.epsilon_start, self.epsilon_end, self.epsilon_anneal_time)
            state = env.state
            action = self.choose_action(q_net, state, epsilon)
            action_ = self.action_space[action] if self.is_single_agent else action
            next_state, cost = env.perform_action(action_, self.is_stochastic_dependence)

            buffer.push(state, np.atleast_1d(action), np.atleast_1d(-cost/self.reward_normalization), next_state)

            if step % self.policy_update_freq == 0:
                self.update_policy(step)

            cost_holder.append(cost)

            if step % self.log_freq == 0:
                cost_rate = sum(cost_holder) / len(cost_holder)
                list_step.append(step)
                list_cost_rate.append(cost_rate)
                path_ = 'stochastic' if self.is_stochastic_dependence else 'no_stochastic'
                self.save_policy(q_net, step, path_model.joinpath(path_))

                # Print log
                current_lr = lr_scheduler.get_last_lr()[-1]
                print(f'Step: {step}, cost rate: {cost_rate: .2f}, lr: {current_lr}, epsilon: {epsilon: .4f}')

        ending_time = datetime.datetime.now()
        training_time = ending_time - starting_time
        print(f"Training time: {training_time}")

        path_log_ = path_log if self.is_stochastic_dependence else path_log.joinpath('no_stochastic')

        torch.save(list_step, path_log_.joinpath('step.pt'))
        torch.save(list_cost_rate, path_log_.joinpath('cost_rate_train.pt'))
        torch.save(training_time, path_log_.joinpath('training_time.pt'))

        plt.plot(list_step, list_cost_rate)
        plt.show()

    def evaluate_(self, q_net, env: System, path_model: Path, path_log: Path):
        path_log_ = path_log if self.is_stochastic_dependence else path_log.joinpath('no_stochastic')
        path_model_ = path_model.joinpath('stochastic' if self.is_stochastic_dependence else 'no_stochastic')
        log_step = torch.load(path_log_.joinpath('step.pt'))
        log_cost_rate = []
        q_net.eval()

        for step in log_step:
            q_net.load_state_dict(torch.load(path_model_.joinpath(f'policy_{step}.pt')))
            average_cost = np.zeros(self.n_runs)
            for i in range(self.n_runs):
                total_cost = 0.0
                env.reset()
                for _ in range(self.episode_len_eval):
                    state = env.state
                    action = self.choose_action(q_net, state, 0.0)
                    action_ = self.action_space[action] if self.is_single_agent else action
                    _, cost = env.perform_action(action_, is_stochastic_dependence=True)
                    total_cost += cost
                average_cost[i] = total_cost / self.episode_len_eval

            mean_cost_rate = np.mean(average_cost)
            log_cost_rate.append(mean_cost_rate)
            print(f'step: {step}, cost_rate: {mean_cost_rate: .3f}')

        torch.save(log_cost_rate, path_log_.joinpath('cost_rate.pt'))

        plt.plot(log_step, log_cost_rate)
        plt.show()

    def compute_q_branch(self, q_branch_net, state, action, masking=False):
        q_all = q_branch_net(state)
        q_branch = []
        for i in range(self.n_components):
            q = q_all[i]
            if masking:
                mask = torch.stack([self.action_mask[int(local_s)] for local_s in state[:, i]], dim=0)
                q += mask

            q_branch.append(q.gather(1, torch.unsqueeze(action[:, i], dim=1)))

        return torch.cat(q_branch, dim=1)

    def compute_a_max(self, q_net, state, masking=False):
        q_all = q_net(state).detach()
        action_max = []
        for i in range(self.n_components):
            q = q_all[i]
            if masking:
                mask = torch.stack([self.action_mask[int(local_s)] for local_s in state[:, i]], dim=0)
                q += mask

            action_max.append(torch.argmax(q, dim=1, keepdim=True))

        return torch.cat(action_max, dim=1)








