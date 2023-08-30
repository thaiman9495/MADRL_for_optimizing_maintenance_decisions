import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, shared_conf, value_conf, advantage_conf):
        """
        This class represents the dueling Q-network

        Args:
            n_inputs: number of inputs
            n_outputs: number of ouputs
            shared_conf: a list containing number of neurons in each hidden layer of the shared module
            value_conf: a list containing number of neurons in each hidden layer of the value module
            advantage_conf: a list containing number of neurons in each hidden layer of the advantage module
        """

        super().__init__()
        self.n_layers_shared = len(shared_conf)
        self.n_layers_value = len(value_conf)
        self.n_layers_advantage = len(advantage_conf)
        self.m_shared = nn.ModuleList()
        self.m_value = nn.ModuleList()
        self.m_advantage = nn.ModuleList()

        # Shared module
        self.m_shared.append(nn.Linear(n_inputs, shared_conf[0]))
        for i in range(1, self.n_layers_shared):
            self.m_shared.append(nn.Linear(shared_conf[i-1], shared_conf[i]))

        # Value module
        self.m_value.append(nn.Linear(shared_conf[-1], value_conf[0]))
        for i in range(1, self.n_layers_value):
            self.m_value.append(nn.Linear(value_conf[i-1], value_conf[i]))
        self.m_value.append(nn.Linear(value_conf[-1], 1))

        # Advantage module
        self.m_advantage.append(nn.Linear(shared_conf[-1], advantage_conf[0]))
        for i in range(1, self.n_layers_advantage):
            self.m_advantage.append(nn.Linear(advantage_conf[i-1], advantage_conf[i]))
        self.m_advantage.append(nn.Linear(advantage_conf[-1], n_outputs))

    def forward(self, state):
        # Shared module
        x = F.relu(self.m_shared[0](state))
        for i in range(1, self.n_layers_shared):
            x = F.relu(self.m_shared[i](x))

        # Value module
        v = F.relu(self.m_value[0](x))
        for i in range(1, self.n_layers_value):
            v = F.relu(self.m_value[i](v))
        v = self.m_value[-1](v)

        # Advantage module
        a = F.relu(self.m_advantage[0](x))
        for i in range(1, self.n_layers_advantage):
            a = F.relu(self.m_advantage[i](a))
        a = self.m_advantage[-1](a)

        # Output action-value function
        a_mean = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_mean

        return q
