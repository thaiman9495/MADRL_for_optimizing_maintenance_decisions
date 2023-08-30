import torch
import torch.nn as nn
import torch.nn.functional as F


class QmixMixer(nn.Module):
    def __init__(self, n_agents, n_neurons_mixer, n_neurons_hyper):
        super().__init__()
        self.n_agents = n_agents
        self.n_neurons_mixer = n_neurons_mixer
        self.n_neurons_hyper = n_neurons_hyper

        # Weights
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, self.n_agents * self.n_neurons_mixer))

        self.hyper_w_2 = nn.Sequential(nn.Linear(self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, self.n_neurons_mixer))

        # Bias
        self.hyper_b_1 = nn.Linear(self.n_agents, self.n_neurons_mixer)
        # self.hyper_b_1 = nn.Sequential(nn.Linear(self.n_agents, self.n_neurons_hyper),
        #                                nn.ReLU(),
        #                                nn.Linear(self.n_neurons_hyper, self.n_neurons_mixer))

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, 1))

    def forward(self, qs, states):
        qs_ = qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.n_neurons_mixer)
        b1 = b1.view(-1, 1, self.n_neurons_mixer)
        hidden = F.elu(torch.bmm(qs_, w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w_2(states))
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.n_neurons_mixer, 1)
        b2 = b2.view(-1, 1, 1)

        # Compute Q_tot
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.squeeze(1)

        return q_tot


class QmixMixerDecentralized(nn.Module):
    def __init__(self, n_agents, n_actions, n_neurons_mixer, n_neurons_hyper):
        super().__init__()
        self.n_agents = n_agents
        self.n_neurons_mixer = n_neurons_mixer
        self.n_neurons_hyper = n_neurons_hyper

        # Weights
        self.hyper_w_1 = nn.Sequential(nn.Linear((1 + n_actions) * self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, self.n_agents * self.n_neurons_mixer))

        self.hyper_w_2 = nn.Sequential(nn.Linear((1 + n_actions) * self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, self.n_neurons_mixer))

        # Bias
        self.hyper_b_1 = nn.Linear((1 + n_actions) * self.n_agents, self.n_neurons_mixer)
        # self.hyper_b_1 = nn.Sequential(nn.Linear(self.n_agents, self.n_neurons_hyper),
        #                                nn.ReLU(),
        #                                nn.Linear(self.n_neurons_hyper, self.n_neurons_mixer))

        self.hyper_b_2 = nn.Sequential(nn.Linear((1 + n_actions) * self.n_agents, self.n_neurons_hyper),
                                       nn.ReLU(),
                                       nn.Linear(self.n_neurons_hyper, 1))

    def forward(self, qs, states):
        qs_ = qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.n_neurons_mixer)
        b1 = b1.view(-1, 1, self.n_neurons_mixer)
        hidden = F.elu(torch.bmm(qs_, w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w_2(states))
        b2 = self.hyper_b_2(states)
        w2 = w2.view(-1, self.n_neurons_mixer, 1)
        b2 = b2.view(-1, 1, 1)

        # Compute Q_tot
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.squeeze(1)

        return q_tot
