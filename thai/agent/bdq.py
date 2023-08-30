import copy
import torch
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from thai.agent.parent import Agent
from thai.model.braching_net import BrachingNet
from thai.buffer.uniform_buffer import UniformBuffer


class Bdq(Agent):
    def __init__(self, n_components, n_c_actions, n_c_states, device, **param):
        super().__init__(n_components, n_c_actions, n_c_states, device, is_single_agent=False, **param)

        # Training network
        p = n_components, n_components, n_c_actions, param['shared_conf'], param['value_conf'], param['advantage_conf']
        self.q_tot_net = BrachingNet(*p).to(device)

        # Target network
        self.q_tot_net_target = copy.deepcopy(self.q_tot_net)

        # Set target networks in evaluation mode and copy weights from learning networks into them
        self.q_tot_net_target.eval()
        self.update_target_net()

        # Optimizer
        lr_start = param['lr_start']
        self.optimizer = Adam([
            {'params': self.q_tot_net.m_shared.parameters(), 'lr': lr_start / np.sqrt(self.n_components + 1)},
            {'params': self.q_tot_net.m_value.parameters(), 'lr': lr_start},
            {'params': self.q_tot_net.m_advantage.parameters(), 'lr': lr_start}])

        # Learning rate scheduling
        self.lr_scheduler = StepLR(self.optimizer, param['lr_step_size'], param['lr_decay_constant'])

        # Buffer intialization
        self.buffer = UniformBuffer(self.buffer_capacity, self.n_components, self.batch_size, is_action_index=False)

    def update_target_net(self):
        self.q_tot_net_target.load_state_dict(self.q_tot_net.state_dict())

    def update_policy(self, step: int):
        if len(self.buffer) > self.batch_size:
            # Sample a mini batch from relay buffer
            state, action, reward, next_state = self.buffer.sample(self.device)

            # Compute q_branch(s_k, a_k)
            q_branch = self.compute_q_branch(self.q_tot_net, state, action, masking=False)

            # Compute action_max in next timestamp
            action_max = self.compute_a_max(self.q_tot_net, next_state, masking=True)

            # Compute q_branch_next
            q_branch_next = self.compute_q_branch(self.q_tot_net_target, next_state, action_max, masking=True)

            # Compute target
            target = reward + self.gamma * torch.sum(q_branch_next, dim=1, keepdim=True) / self.n_components
            target = target.detach()

            # Compute loss and update policy
            loss_branch = (q_branch - target) ** 2 / self.n_components
            loss = torch.sum(loss_branch) / self.batch_size

            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradient
            clip_grad_norm_(self.q_tot_net.parameters(), max_norm=self.grad_norm_clip)

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

