import torch
import numpy as np

from thai.utility.episode_recorder import EpisodeRecoder
from thai.utility.raw_sequence import RawSequence
from thai.buffer.episode_buffer import EpisodeBuffer

n_episodes = 100
episode_length = 10
batch_size = 3
state_dim = 5
n_actions = 3

state_raw_sq = [i * np.ones(state_dim) for i in range(episode_length)]
action_raw_sq = [np.random.randint(n_actions, size=state_dim) for i in range(episode_length)]
reward_raw_sq = [i + 1 for i in range(episode_length)]
next_state_raw_sq = [(i + 1) * np.ones(state_dim) for i in range(episode_length)]

raw_sq = RawSequence()
for state, action, reward, next_state in zip(state_raw_sq, action_raw_sq, reward_raw_sq, next_state_raw_sq):
    raw_sq.push(state, action, reward, next_state)

# Create list of episodes
list_episode = [EpisodeRecoder(raw_sq, state_value_max=3.0, reward_normalization=10.0, n_local_actions=3)
                for _ in range(n_episodes)]

# Create an episode buffer
buffer = EpisodeBuffer(capacity=n_episodes, batch_size=batch_size, episode_len=episode_length)

# Put episodes into the buffer
for episode in list_episode:
    buffer.push(episode)

# Sample a mini-batch from the buffer
device = torch.device('cpu')
batch_observation_sq, batch_action_sq, batch_reward_sq, batch_next_observation_sq, batch_global_state_sq, batch_next_global_state_sq = buffer.sample(device)

print(batch_action_sq)

# Print batch state
# print(batch_observation_sq)
timestamp = 2
agent = 3
print('')
print(f'batch of observations of agent {agent} at timestamp {timestamp}')
print(batch_observation_sq[timestamp, agent-1])

# # Print batch state
# print(batch_next_state_raw_sq)
# timestamp = 0
# agent = 2
# print('')
# print(f'batch of raw next states of agent {agent} at timestamp {timestamp}')
# print(batch_next_state_raw_sq[timestamp, agent-1])

# # Print batch state
# print(batch_state_sp.shape)
# timestamp = 0
# agent = 2
# print('')
# print(f'batch of states of agen {agent} at timestamp {timestamp}')
# print(batch_state_sp[timestamp, agent-1])

# # Print batch action
# print(batch_action_sp)
# timestamp = 0
# agent = 2
# print('')
# print(f'batch of actions of agen {agent} at timestamp {timestamp}')
# print(batch_action_sp[timestamp, agent-1])

# # Print batch reward
# print(batch_reward_sq)
# episode_n = 1
# print('')
# print(f'reward sequence of episode {episode_n}')
# print(batch_reward_sq[episode_n-1])

# Print next state
# print(batch_next_state_sp.shape)
# timestamp = 0
# agent = 2
# print('')
# print(f'batch of next states of agent {agent} at timestamp {timestamp}')
# print(batch_next_state_sp[timestamp, agent-1])
