import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnAgent(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2):
        super().__init__()
        self.n_layers_1 = len(hidden_conf_1)
        self.n_layers_2 = len(hidden_conf_2)
        self.n_hidden_rnn = rnn_conf

        # Initialize MLP 1
        self.mpl_1 = nn.ModuleList()
        self.mpl_1.append(nn.Linear(n_inputs, hidden_conf_1[0]))
        for i in range(1, self.n_layers_1):
            self.mpl_1.append(nn.Linear(hidden_conf_1[i-1], hidden_conf_1[i]))

        # Initialize recurrent network
        self.rnn = nn.GRUCell(hidden_conf_1[-1], self.n_hidden_rnn)

        # Initialize MLP 2
        self.mpl_2 = nn.ModuleList()
        self.mpl_2.append(nn.Linear(self.n_hidden_rnn, hidden_conf_2[0]))
        for i in range(1, self.n_layers_2):
            self.mpl_2.append(nn.Linear(hidden_conf_2[i-1], hidden_conf_2[i]))
        self.mpl_2.append(nn.Linear(hidden_conf_2[-1], n_outputs))

    def initialize_hidden(self, batch_size, device=torch.device('cpu')):
        return torch.zeros(batch_size, self.n_hidden_rnn).to(device)

    def forward(self, inputs, hidden_state_in):
        # MLP 1
        x = F.relu(self.mpl_1[0](inputs))
        for i in range(1, self.n_layers_1):
            x = F.relu(self.mpl_1[i](x))

        # Recrrent network
        hidden_state = self.rnn(x, hidden_state_in)

        # MLP 2
        x = F.relu(self.mpl_2[0](hidden_state))
        for i in range(1, self.n_layers_2):
            x = F.relu(self.mpl_2[i](x))
        q = self.mpl_2[-1](x)

        return q, hidden_state





