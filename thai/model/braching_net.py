import torch
import torch.nn as nn
import torch.nn.functional as F


class BrachingNet(nn.Module):
    def __init__(self, n_inputs, n_branches, n_outputs, shared_conf, value_conf, advantage_conf):
        """
        This class represents the braching dueling Q-network

        Args:
            n_inputs: number of inputs
            n_branches: number of branches
            n_outputs: number of ouputs in each branch
            shared_conf: a list containing number of neurons in each hidden layer of the shared module
            value_conf: a list containing number of neurons in each hidden layer of the value module
            advantage_conf: a list containing number of neurons in each hidden layer of the advantage module
        """

        super().__init__()
        self.n_layers_shared = len(shared_conf)
        self.n_layers_value = len(value_conf)
        self.n_layers_advantage = len(advantage_conf)

        # Initialize shared module
        self.m_shared = nn.ModuleList()
        self.m_shared.append(nn.Linear(n_inputs, shared_conf[0]))
        for i in range(1, self.n_layers_shared):
            self.m_shared.append(nn.Linear(shared_conf[i-1], shared_conf[i]))

        # Initialize value module
        self.m_value = nn.ModuleList()
        self.m_value.append(nn.Linear(shared_conf[-1], value_conf[0]))
        for i in range(1, self.n_layers_value):
            self.m_value.append(nn.Linear(value_conf[i-1], value_conf[i]))
        self.m_value.append(nn.Linear(value_conf[-1], 1))

        # Initialize advantage modules
        self.m_advantage = nn.ModuleList()
        for _ in range(n_branches):
            self.m_advantage.append(nn.ModuleList())

        for m in self.m_advantage:
            m.append(nn.Linear(shared_conf[-1], advantage_conf[0]))
            for i in range(1, self.n_layers_advantage):
                m.append(nn.Linear(advantage_conf[i-1], advantage_conf[i]))
            m.append(nn.Linear(advantage_conf[-1], n_outputs))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
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
        q_out = []
        for m in self.m_advantage:
            a = F.relu(m[0](x))
            for i in range(1, self.n_layers_advantage):
                a = F.relu(m[i](a))
            a = m[-1](a)
            a_mean = torch.mean(a, dim=1, keepdim=True)
            q = v + a - a_mean
            q_out.append(q)

        q_out_tensor = torch.stack(q_out)
        return q_out_tensor


if __name__ == '__main__':
    # Set training device: GPU or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    my_input = torch.randn(10, 4).to(device)
    my_braching_net = BrachingNet(n_inputs=4,
                                   n_branches=3,
                                   n_outputs=6,
                                   shared_conf=[32, 32],
                                   value_conf=[32, 32],
                                   advantage_conf=[16]).to(device)

    my_output = my_braching_net(my_input)
    print(my_output)
    print(len(my_output))
