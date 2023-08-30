import torch
import numpy as np

from thai.utility.episode_recorder import EpisodeRecoder
from thai.buffer.episode_buffer import EpisodeBuffer
from thai.model.rnn_agent import RnnAgent
from thai.mixer.qmix_mixer import QmixMixer


def compute_q_joint_sq(agent, mixer, state_raw_sq, state_sq, action_sq, hidden_rnn):
    q_joint_sq = []
    for state_raw, state, action in zip(state_raw_sq, state_sq, action_sq):
        selelcted_q_all_agent = []
        for idx, ag in enumerate(agent):
            q_single_agent, hidden_rnn[idx] = ag(state[idx], hidden_rnn[idx])
            selected_q_single_agent = torch.gather(q_single_agent, dim=1, index=action[idx])
            selelcted_q_all_agent.append(selected_q_single_agent)

        selelcted_q_all_agent = torch.cat(selelcted_q_all_agent, dim=1)
        q_joint = mixer(selelcted_q_all_agent, state_raw)
        q_joint_sq.append(q_joint)

    q_joint_sq = torch.cat(q_joint_sq, dim=1)
    return q_joint_sq


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
agents = [RnnAgent(n_inputs, n_outputs, hidden_conf_1, rnn_conf, hidden_conf_2) for i in range(n_agents)]

# Initialize mixer
mixer = QmixMixer(n_agents, n_neurons_mixer=64, n_neurons_hyper=64)

hidden_rnn = [ag.initialize_hidden(batch_size) for ag in agents]

q_joint_sq = compute_q_joint_sq(agents, mixer, state_raw_sq, state_sq, action_sq, hidden_rnn)
print(q_joint_sq)






