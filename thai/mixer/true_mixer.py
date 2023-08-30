import torch
import torch.nn as nn
import torch.nn.functional as F


class TrueMixer(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_conf):
        super().__init__()
        self.n_hidden_layers = len(hidden_conf)
        self.fc = nn.ModuleList()

        self.fc.append(nn.Linear(n_inputs, hidden_conf[0]))
        for i in range(1, self.n_hidden_layers):
            self.fc.append(nn.Linear(hidden_conf[i-1], hidden_conf[i]))
        self.fc.append(nn.Linear(hidden_conf[-1], n_outputs))

    def forward(self, q, state):
        x = F.relu(self.fc[0](torch.cat([q, state], dim=1)))
        for i in range(1, self.n_hidden_layers):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)

        return x


class TrueMixerOld(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_conf):
        super().__init__()
        self.n_hidden_layers = len(hidden_conf)
        self.fc = nn.ModuleList()

        self.fc.append(nn.Linear(n_inputs, hidden_conf[0]))
        for i in range(1, self.n_hidden_layers):
            self.fc.append(nn.Linear(hidden_conf[i-1], hidden_conf[i]))
        self.fc.append(nn.Linear(hidden_conf[-1], n_outputs))

    def forward(self, q, state):
        x = F.relu(self.fc[0](torch.cat([q, state], dim=1)))
        for i in range(1, self.n_hidden_layers):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return x
