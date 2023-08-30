import torch
import numpy as np

from thai.utility.episode_recorder import EpisodeRecoder
from thai.buffer.episode_buffer import EpisodeBuffer
from thai.model.rnn_agent import RnnAgent
from thai.mixer.qmix_mixer import QmixMixer

n_episodes = 100
episode_length = 5
batch_size = 4
state_dim = 5
n_agents = state_dim
n_actions = 3

state_sq = [i * np.ones(state_dim) for i in range(episode_length)]
action_sq = [np.random.randint(n_actions, size=state_dim) for i in range(episode_length)]
reward_sq = [np.atleast_1d(i+1) for i in range(episode_length)]
next_state_sq = [(i+1) * np.ones(state_dim) for i in range(episode_length)]

# Create list of episodes
list_episode = [EpisodeRecoder(state_sq, action_sq, reward_sq, next_state_sq) for _ in range(n_episodes)]

# Create an episode buffer
buffer = EpisodeBuffer(capacity=n_episodes, batch_size=batch_size, episode_len=episode_length)

# Put episodes into the buffer
for episode in list_episode:
    buffer.push(episode)

# Sample a mini-batch from the buffer
device = torch.device('cpu')
sequences = buffer.sample(device)
state_raw_sq, next_state_raw_sq, state_sq, action_sq, reward_sq, next_state_sq = sequences

# Intialize agents
n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2 = 2, 3, [64], 4, [64]
agent = [RnnAgent(n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2) for i in range(n_agents)]

# Initialize mixer
mixer = QmixMixer(n_agents, n_neurons_mixer=64, n_neurons_hyper=64)


hidden_rnn = [ag.initialize_hidden(batch_size) for ag in agent]
a_max_sq = []
for state_raw, state in zip(state_raw_sq, state_sq):
    a_max_all_agent = []
    for idx, ag in enumerate(agent):
        q_single_agent, hidden_rnn[idx] = ag(state[idx], hidden_rnn[idx])
        a_max_single_agent = torch.argmax(q_single_agent, dim=1, keepdim=True)
        a_max_all_agent.append(a_max_single_agent)

    a_max = torch.stack(a_max_all_agent, dim=0)

    a_max_sq.append(a_max)
a_max_sq = torch.stack(a_max_sq, dim=0)

print(a_max_sq)
