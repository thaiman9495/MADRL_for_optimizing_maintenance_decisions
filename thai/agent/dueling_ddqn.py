import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from thai.env.system import System
from thai.agent.parent import Agent
from thai.model.dueling_net import DuelingNet
from thai.buffer.uniform_buffer import UniformBuffer


class DuelingDdqn(Agent):
    def __init__(self, n_components, n_c_actions, n_c_states, device, **param):
        super().__init__(n_components, n_c_actions, n_c_states, device, is_single_agent=True, **param)

        #  Training etwrok
        p = n_components, self.n_s_actions, param['shared_conf'], param['value_conf'], param['advantage_conf']
        self.q_net = DuelingNet(*p).to(self.device)

        # Target netwrok
        self.q_net_target = copy.deepcopy(self.q_net)

        # Set target networks in evaluation mode and copy weights from learning networks into them
        self.q_net_target.eval()
        self.update_target_net()

        # Optimizers
        lr_start = param['lr_start']
        self.optimizer = optim.Adam([
            {'params': self.q_net.m_shared.parameters(), 'lr': lr_start / 2.0},
            {'params': self.q_net.m_value.parameters(), 'lr': lr_start},
            {'params': self.q_net.m_advantage.parameters(), 'lr': lr_start}])

        # Learning rate scheduling
        self.lr_scheduler = StepLR(self.optimizer, param['lr_step_size'], param['lr_decay_constant'])

        # Buffer
        self.buffer = UniformBuffer(self.buffer_capacity, self.n_components, self.batch_size, is_action_index=True)

    def update_target_net(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

    def update_policy(self, step):
        if len(self.buffer) > self.batch_size:
            # Sample a mini batch from relay buffer
            batch_state, batch_action, batch_reward, batch_next_state = self.buffer.sample(self.device)

            # Compute q
            q = self.q_net(batch_state).gather(1, batch_action)

            # Compute mask corresponding to "next state"
            mask = [self.action_mask[self.state_to_id[tuple(s.cpu().numpy().astype(int))]] for s in batch_next_state]
            mask = torch.stack(mask, dim=0)

            # Compute action max
            q_temp = self.q_net(batch_next_state).detach() + mask
            action_max = torch.argmax(q_temp, 1).unsqueeze(1)

            # Compute q_target
            q_temp = self.q_net_target(batch_next_state) + mask
            q_target = batch_reward + self.gamma * q_temp.gather(1, action_max)

            # Compute loss
            error = q - q_target.detach()
            loss = torch.sum(error ** 2) / self.batch_size

            # Update policy network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()

            # Schedueling learning rate
            if self.lr_scheduler.get_last_lr()[-1] > self.lr_end:
                self.lr_scheduler.step()

            # Update target networks (hard update)
            if step % self.target_update_freq == 0:
                self.update_target_net()

    def train(self, env: System, path_model, path_log):
        self.train_(self.q_net, self.buffer, self.lr_scheduler, env, path_model, path_log)

    def evaluate(self, env: System, path_model, path_log):
        self.evaluate_(self.q_net, env, path_model, path_log)

