import copy
import torch
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

from thai.env.system import System
from thai.model.rnn_agent import RnnAgent
from thai.mixer.qmix_mixer import QmixMixerDecentralized
from thai.mixer.true_mixer import TrueMixer
from thai.utility.episode_recorder import EpisodeRecoder
from thai.buffer.episode_buffer import EpisodeBuffer
from thai.utility.action_mask import create_action_mask
from thai.utility.raw_sequence import RawSequence
from thai.param.wqmix import WqmixParam


class Wqmix:
    def __init__(self, device, param: WqmixParam):
        self.device = device
        self.param = param

        self.action_mask = create_action_mask(int(param.input_max + 1), param.n_actions).to(device)
        self.agent_id = np.eye(param.n_agents)
        self.action_encoded = np.eye(param.n_actions)

        # training agents
        p = 1 + param.n_actions + param.n_agents, param.n_actions, param.hidden_conf_1, param.rnn_conf, param.hidden_conf_2
        self.agent_tot = RnnAgent(*p).to(device)
        self.agent_true = RnnAgent(*p).to(device)
        self.mixer_tot = QmixMixerDecentralized(param.n_agents, param.n_actions, param.n_neurons_mixer, param.n_neurons_hyper).to(device)
        self.mixer_true = TrueMixer(param.n_agents * (2 + param.n_actions), 1, param.mixer_true_conf).to(device)

        # target agents
        self.agent_tot_target = copy.deepcopy(self.agent_tot)
        self.agent_true_target = copy.deepcopy(self.agent_true)
        self.mixer_tot_target = copy.deepcopy(self.mixer_tot)
        self.mixer_true_target = copy.deepcopy(self.mixer_true)

        self.put_target_networks_into_evaluation_mode()
        self.update_target_agents()

        self.optimizer = Adam([{'params': self.agent_tot.parameters(), 'lr': param.lr_start},
                               {'params': self.agent_true.parameters(), 'lr': param.lr_start},
                               {'params': self.mixer_true.parameters(), 'lr': param.lr_start},
                               {'params': self.mixer_tot.parameters(), 'lr': param.lr_start}])

        self.lr_scheduler = StepLR(self.optimizer, param.lr_step_size, param.lr_decay_constant)
        self.buffer = EpisodeBuffer(param.buffer_capacity, param.batch_size, param.episode_len_train)

    def update_target_agents(self):
        self.agent_tot_target.load_state_dict(self.agent_tot.state_dict())
        self.agent_true_target.load_state_dict(self.agent_true.state_dict())
        self.mixer_tot_target.load_state_dict(self.mixer_tot.state_dict())
        self.mixer_true_target.load_state_dict(self.mixer_true.state_dict())

    def put_target_networks_into_evaluation_mode(self):
        self.agent_tot_target.eval()
        self.agent_true_target.eval()
        self.mixer_tot_target.eval()
        self.mixer_true_target.eval()

    def clip_gradient(self):
        clip_grad_norm_(self.agent_tot.parameters(), max_norm=self.param.grad_norm_clip)
        clip_grad_norm_(self.agent_true.parameters(), max_norm=self.param.grad_norm_clip)
        clip_grad_norm_(self.mixer_tot.parameters(), max_norm=self.param.grad_norm_clip)
        clip_grad_norm_(self.mixer_true.parameters(), max_norm=self.param.grad_norm_clip)

    def save_policy(self, step, path: pathlib.Path):
        saved_state_dict = self.agent_tot.state_dict()
        torch.save(saved_state_dict, path.joinpath(f'policy_{step}.pt'))

    @staticmethod
    def exploration_shudeler(step, epsilon_start: float, epsilon_end: float, epsilon_anneal_time: int):
        if step <= epsilon_anneal_time:
            return step * (epsilon_end - epsilon_start) / epsilon_anneal_time + epsilon_start
        return epsilon_end

    def choose_action(self, state, hidden_rnn_in, epsilon):
        """
        Decentralized agents choose its own action to form the joint action

        Args:
            state: combination of current state and last action [s_k, a_{k-1}]
            hidden_rnn_in: hidden features of recurrent network from last step
            epsilon: exploration constant

        Returns: choosen joint action
        """

        # --------------------------------------------------------------------------------------------------------------
        # Choose action randomly
        # --------------------------------------------------------------------------------------------------------------
        random_joint_action = []
        for idx, s_ in enumerate(state[:, 0]):
            s = int(s_)
            if s == 0:
                a = 0
            else:
                if s == 3:
                    a = np.random.choice([0, 2])
                else:
                    a = np.random.randint(self.param.n_actions)
            random_joint_action.append(a)

        random_joint_action = np.array(random_joint_action, dtype=int)

        # --------------------------------------------------------------------------------------------------------------
        # Choose action according to learned policies
        # --------------------------------------------------------------------------------------------------------------

        # Convert to tensor and normalize
        state_t = torch.FloatTensor(state).to(self.device)
        state_t[:, 0] /= self.param.input_max

        learned_joint_action = []
        hidden_rnn_out = []

        for i in range(self.param.n_agents):
            with torch.no_grad():
                q, hidden_rnn = self.agent_tot(torch.unsqueeze(state_t[i], dim=0), hidden_rnn_in[i])
            q += self.action_mask[int(state[i, 0])]
            learned_joint_action.append(torch.argmax(q, dim=1, keepdim=False))
            hidden_rnn_out.append(hidden_rnn)

        learned_joint_action = np.array(torch.cat(learned_joint_action, dim=0).detach().cpu())

        if np.random.uniform() < epsilon:
            return random_joint_action, hidden_rnn_out
        else:
            return learned_joint_action, hidden_rnn_out

    def compute_q_joint_sq(self, agent, mixer, observation_sq, action_sq, global_state_sq, masking=False):
        q_joint_sq = []
        hidden_rnn = [agent.initialize_hidden(self.param.batch_size, self.device) for _ in range(self.param.n_agents)]
        for observation, action, global_state in zip(observation_sq, action_sq, global_state_sq):
            selelcted_q_all_agent = []
            for i in range(self.param.n_agents):
                q_single_agent, hidden_rnn[i] = agent(observation[i], hidden_rnn[i])
                if masking:
                    ob_integer = (observation[i, :, 0] * self.param.input_max).long()
                    mask = torch.stack([self.action_mask[s] for s in ob_integer], dim=0)
                    q_single_agent += mask

                selected_q_single_agent = torch.gather(q_single_agent, dim=1, index=action[i])
                selelcted_q_all_agent.append(selected_q_single_agent)

            selelcted_q_all_agent = torch.cat(selelcted_q_all_agent, dim=1)
            q_joint = mixer(selelcted_q_all_agent, global_state)
            q_joint_sq.append(q_joint)

        q_joint_sq = torch.cat(q_joint_sq, dim=1)
        return q_joint_sq

    def compute_action_max_sq(self, agent, observation_sq, masking=False):
        hidden_rnn = [agent.initialize_hidden(self.param.batch_size, self.device) for _ in range(self.param.n_agents)]
        a_max_sq = []
        for observation in observation_sq:
            a_max_all_agent = []
            for i in range(self.param.n_agents):
                q_single_agent, hidden_rnn[i] = agent(observation[i], hidden_rnn[i])
                if masking:
                    ob_integer = (observation[i, :, 0] * self.param.input_max).long()
                    mask = torch.stack([self.action_mask[s] for s in ob_integer], dim=0)
                    q_single_agent = q_single_agent.detach()
                    q_single_agent += mask

                a_max_single_agent = torch.argmax(q_single_agent, dim=1, keepdim=True)
                a_max_all_agent.append(a_max_single_agent)

            a_max = torch.stack(a_max_all_agent, dim=0)
            a_max_sq.append(a_max)
        a_max_sq = torch.stack(a_max_sq, dim=0)

        return a_max_sq

    def update_policy(self, step):
        if len(self.buffer) > self.param.batch_size:
            # Sample a mini batch from relay buffer
            ob_sq, a_sq, r_sq, next_ob_sq, global_s_sq, next_global_s_sq = self.buffer.sample(self.device)

            q_tot_sq = self.compute_q_joint_sq(self.agent_tot, self.mixer_tot, ob_sq, a_sq, global_s_sq)
            q_true_sq = self.compute_q_joint_sq(self.agent_true, self.mixer_true, ob_sq, a_sq, global_s_sq)
            a_max_sq = self.compute_action_max_sq(self.agent_tot, next_ob_sq, masking=True)
            q_true_next_sq = self.compute_q_joint_sq(self.agent_true_target, self.mixer_true_target, next_ob_sq,
                                                     a_max_sq, next_global_s_sq, masking=True)

            # Compute target
            target_sq = r_sq + self.param.discount_factor * q_true_next_sq
            target_sq = target_sq.detach()

            # Compute weights
            w_s = torch.ones_like(target_sq) * self.param.weighting_constant
            one_s = torch.ones_like(target_sq)
            weights = torch.where(q_tot_sq < target_sq, one_s, w_s)

            # Compute losses and update policy networkd
            error_tot = q_tot_sq - target_sq
            error_true = q_true_sq - target_sq
            loss_tot = torch.sum(weights * error_tot ** 2) / self.param.batch_size
            loss_true = torch.sum(error_true ** 2) / self.param.batch_size

            self.optimizer.zero_grad()
            loss_tot.backward()
            loss_true.backward()

            self.clip_gradient()
            self.optimizer.step()

            if self.lr_scheduler.get_last_lr()[-1] > self.param.lr_end:
                self.lr_scheduler.step()

            if step % self.param.target_update_freq == 0:
                self.update_target_agents()

    def initialize_training(self):
        last_action = np.zeros(shape=(self.param.n_agents, self.param.n_actions))
        last_action[:, 0] = 1.0
        hidden_rnn = [self.agent_tot.initialize_hidden(1, self.device) for _ in range(self.param.n_agents)]
        raw_sq = RawSequence()
        return raw_sq, hidden_rnn, last_action

    def train(self, env: System, path_model: pathlib.Path, path_log: pathlib.Path):
        # Prepare
        cost_holder = deque(maxlen=5000)
        list_cost_rate = []
        list_step = []

        raw_sq, hidden_rnn, last_action = self.initialize_training()
        env.reset()

        # Main training loop
        starting_time = datetime.datetime.now()
        for step in range(1, self.param.n_steps_training + 1):
            p_ep = step, self.param.epsilon_start, self.param.epsilon_end, self.param.epsilon_anneal_time
            epsilon = self.exploration_shudeler(*p_ep)

            state = env.state
            state_in = np.concatenate([state.reshape(-1, 1), last_action, self.agent_id], axis=1)
            action, hidden_rnn = self.choose_action(state_in, hidden_rnn, epsilon)
            next_state, cost = env.perform_action(action, self.param.is_stochastic_dependence)

            raw_sq.push(state, action, -cost, next_state)

            if step % self.param.episode_len_train == 0:
                p_buffer = raw_sq, self.param.input_max, self.param.reward_normalization, self.param.n_actions
                self.buffer.push(EpisodeRecoder(*p_buffer))
                env.reset()
                raw_sq, hidden_rnn, last_action = self.initialize_training()
            else:
                last_action = np.array([self.action_encoded[int(a)] for a in action])

            # Update policy
            if step % self.param.policy_update_freq == 0:
                self.update_policy(step)

            # Print log
            cost_holder.append(cost)

            if step % self.param.log_freq == 0:
                cost_rate = sum(cost_holder) / len(cost_holder)
                list_step.append(step)
                list_cost_rate.append(cost_rate)
                path_ = 'stochastic' if self.param.is_stochastic_dependence else 'no_stochastic'
                self.save_policy(step, path_model.joinpath(path_))

                current_lr = self.lr_scheduler.get_last_lr()[-1]
                print(f'Step: {step}, cost rate: {cost_rate: .2f}, lr: {current_lr}, epsilon: {epsilon: .4f}')

        ending_time = datetime.datetime.now()
        training_time = ending_time - starting_time
        print(f"Training time: {training_time}")

        path_log_ = path_log if self.param.is_stochastic_dependence else path_log.joinpath('no_stochastic')

        torch.save(list_step, path_log_.joinpath('step.pt'))
        torch.save(list_cost_rate, path_log_.joinpath('cost_rate_train.pt'))
        torch.save(training_time, path_log_.joinpath('training_time.pt'))

        plt.plot(list_step, list_cost_rate)
        plt.show()

    def evaluate(self, env: System, path_model: pathlib.Path, path_log: pathlib.Path):
        path_log_ = path_log if self.param.is_stochastic_dependence else path_log.joinpath('no_stochastic')
        path_model_ = path_model.joinpath('stochastic' if self.param.is_stochastic_dependence else 'no_stochastic')
        log_step = torch.load(path_log_.joinpath('step.pt'))

        self.agent_tot.eval()
        log_cost_rate = []
        component_id = np.arange(self.param.n_agents)
        average_cost = np.zeros(self.param.n_runs)

        for step in log_step:
            policy = torch.load(path_model_.joinpath(f'policy_{step}.pt'))
            self.agent_tot.load_state_dict(policy)

            for i in range(self.param.n_runs):
                total_cost = 0.0
                env.reset()
                last_action = np.zeros(shape=(self.param.n_agents, self.param.n_actions))
                last_action[:, 0] = 1.0
                hidden_rnn = [self.agent_tot.initialize_hidden(1, self.device) for _ in range(self.param.n_agents)]
                for _ in range(self.param.episode_len_eval):
                    state = env.state
                    state_in = np.concatenate([state.reshape(-1, 1), last_action, self.agent_id], axis=1)
                    action, hidden_rnn = self.choose_action(state_in, hidden_rnn, 0.0)
                    _, cost = env.perform_action(action, self.param.is_stochastic_dependence)
                    total_cost += cost
                    last_action = np.array([self.action_encoded[int(a)] for a in action])

                average_cost[i] = total_cost / self.param.episode_len_eval

            mean_cost_rate = np.mean(average_cost)
            log_cost_rate.append(mean_cost_rate)
            print(f'step: {step}, cost_rate: {mean_cost_rate: .3f}')

        torch.save(log_cost_rate, path_log_.joinpath('cost_rate.pt'))

        plt.plot(log_step, log_cost_rate)
        plt.show()













