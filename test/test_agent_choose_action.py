import torch
from thai.model.rnn_agent import RnnAgent

# 3 agents, a batch of 4 episodes
batch_state = torch.FloatTensor([[[1, 2], [3, 4], [5, 6], [7, 8]],
                                 [[7, 8], [9, 1], [2, 3], [9, 5]],
                                 [[4, 5], [6, 7], [8, 9], [6, 7]]])

# batch_state = torch.FloatTensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])

# Initialize agents
n_agents = len(batch_state)
n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2 = 2, 3, [64], 4, [64]

agents = [RnnAgent(n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2) for i in range(n_agents)]

# Choose joint action
hidden_rnn = [agent.initialize_hidden(batch_size=len(batch_state[0])) for agent in agents]
# print(hidden_rnn)
joint_action = []
for idx, agent in enumerate(agents):
    q, hidden_rnn[idx] = agent(batch_state[idx], hidden_rnn[idx])
    joint_action.append(torch.argmax(q, dim=1, keepdim=False))

joint_action = torch.stack(joint_action, dim=1)
# print(hidden_rnn)
print('joint action\n', joint_action)
