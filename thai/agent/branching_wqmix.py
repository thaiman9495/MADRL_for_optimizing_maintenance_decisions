import copy
import torch
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from thai.agent.parent import Agent
from thai.model.braching_net import BrachingNet
from thai.mixer.qmix_mixer import QmixMixer
from thai.buffer.uniform_buffer import UniformBuffer
from thai.model.mpl import Mlp


class BranchingWqmix(Agent):
    def __init__(self, n_components, n_c_actions, n_c_states, device, **param):
        super().__init__(n_components, n_c_actions, n_c_states, device, is_single_agent=False, **param)

        # Weighting constant
        self.w = param['weighting_constant']

        # Training network
        p = n_components, n_components, n_c_actions, param['shared_conf'], param['value_conf'], param['advantage_conf']
        self.q_tot_net = BrachingNet(*p).to(device)
        self.q_true_net = BrachingNet(*p).to(device)
        self.mixer_tot = QmixMixer(n_components, param['n_neurons_mixer'], param['n_neurons_hyper']).to(device)
        self.mixer_true = Mlp(self.n_components * 2, 1, param['mixer_true_conf']).to(device)

        # Target network
        self.q_tot_net_target = copy.deepcopy(self.q_tot_net)
        self.q_true_net_target = copy.deepcopy(self.q_true_net)
        self.mixer_tot_target = copy.deepcopy(self.mixer_tot)
        self.mixer_true_target = copy.deepcopy(self.mixer_true)

        # Set target networks in evaluation mode and copy weights from learning networks into them
        self.q_tot_net_target.eval()
        self.q_true_net_target.eval()
        self.mixer_tot_target.eval()
        self.mixer_true_target.eval()
        self.update_target_net()

        # Optimizer
        lr_start = param['lr_start']
        self.optimizer = Adam([
            {'params': self.q_tot_net.m_shared.parameters(), 'lr': lr_start / np.sqrt(self.n_components + 1)},
            {'params': self.q_tot_net.m_value.parameters(), 'lr': lr_start},
            {'params': self.q_tot_net.m_advantage.parameters(), 'lr': lr_start},
            {'params': self.q_true_net.m_shared.parameters(), 'lr': lr_start / np.sqrt(self.n_components + 1)},
            {'params': self.q_true_net.m_value.parameters(), 'lr': lr_start},
            {'params': self.q_true_net.m_advantage.parameters(), 'lr': lr_start},
            {'params': self.mixer_tot.parameters(), 'lr': lr_start},
            {'params': self.mixer_true.parameters(), 'lr': lr_start}])

        # Learning rate scheduling
        self.lr_scheduler = StepLR(self.optimizer, param['lr_step_size'], param['lr_decay_constant'])

        # Buffer intialization
        self.buffer = UniformBuffer(self.buffer_capacity, self.n_components, self.batch_size, is_action_index=False)

    def update_target_net(self):
        self.q_tot_net_target.load_state_dict(self.q_tot_net.state_dict())
        self.q_true_net_target.load_state_dict(self.q_true_net.state_dict())
        self.mixer_tot_target.load_state_dict(self.mixer_tot.state_dict())
        self.mixer_true_target.load_state_dict(self.mixer_true.state_dict())

    def update_policy(self, step):
        if len(self.buffer) > self.batch_size:
            # Sample a mini batch from relay buffer
            state, action, reward, next_state = self.buffer.sample(self.device)

            # Compute q_tot(s_k, a_k)
            q_tot_branch = self.compute_q_branch(self.q_tot_net, state, action, masking=False)
            q_tot = self.mixer_tot(q_tot_branch, state)

            # Compute q_true(s_k, a_k)
            q_true_branch = self.compute_q_branch(self.q_true_net, state, action)
            q_true = self.mixer_true(torch.cat((q_true_branch, state), dim=1))

            # Compute action_max in next timestamp
            action_max = self.compute_a_max(self.q_tot_net, next_state, masking=True)

            # Compute q_tot_next
            q_true_branch_next = self.compute_q_branch(self.q_true_net_target, next_state, action_max, masking=True)
            q_true_next = self.mixer_true_target(torch.cat((q_true_branch_next, next_state), dim=1))

            # Compute target
            target = reward + self.gamma * q_true_next
            target = target.detach()

            # Compute weights
            w_s = torch.ones_like(target) * self.w
            one_s = torch.ones_like(target)
            weights = torch.where(q_tot < target, one_s, w_s)

            # Compute losses and update policy networkd
            error_tot = q_tot - target
            error_true = q_true - target

            loss_tot = torch.sum(weights * error_tot ** 2) / self.batch_size
            loss_true = torch.sum(error_true ** 2) / self.batch_size

            # loss = loss_tot + loss_true

            self.optimizer.zero_grad()
            loss_tot.backward()
            loss_true.backward()
            # loss.backward()

            # Clip gradient
            clip_grad_norm_(self.q_tot_net.parameters(), max_norm=self.grad_norm_clip)
            clip_grad_norm_(self.q_true_net.parameters(), max_norm=self.grad_norm_clip)
            clip_grad_norm_(self.mixer_tot.parameters(), max_norm=self.grad_norm_clip)
            clip_grad_norm_(self.mixer_true.parameters(), max_norm=self.grad_norm_clip)

            self.optimizer.step()

            if self.lr_scheduler.get_last_lr()[-1] > self.lr_end:
                self.lr_scheduler.step()

            # Update target networks (hard update)
            if step % self.target_update_freq == 0:
                self.update_target_net()

    def train(self, env, path_model, path_log):
        self.train_(self.q_tot_net, self.buffer, self.lr_scheduler, env, path_model, path_log)

    def evaluate(self, env, path_model, path_log):
        self.evaluate_(self.q_tot_net, env, path_model, path_log)




