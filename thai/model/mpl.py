import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_conf):
        super().__init__()
        self.n_hidden_layers = len(hidden_conf)
        self.fc = nn.ModuleList()

        self.fc.append(nn.Linear(n_inputs, hidden_conf[0]))
        for i in range(1, self.n_hidden_layers):
            self.fc.append(nn.Linear(hidden_conf[i-1], hidden_conf[i]))
        self.fc.append(nn.Linear(hidden_conf[-1], n_outputs))

    def forward(self, x_in):
        x = F.relu(self.fc[0](x_in))
        for i in range(1, self.n_hidden_layers):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)

        return x


if __name__ == '__main__':
    # Set training device: GPU or CPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    my_input = torch.randn(15, 4).to(device)
    my_net = Mlp(n_inputs=4, n_outputs=6, hidden_conf=[64, 64]).to(device)

    my_output = my_net(my_input)
    print(my_input)
    print(my_output)
    print(len(my_output))
