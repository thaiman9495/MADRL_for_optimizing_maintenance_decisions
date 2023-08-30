import numpy as np

from thai.utility.raw_sequence import RawSequence
from thai.utility.episode_recorder import EpisodeRecoder

episode_length = 4
state_dim = 2      # number of agents
n_actions = 3
batch_size = 6

state_raw_sq = [i * np.ones(state_dim) for i in range(episode_length)]
action_raw_sq = [np.random.randint(n_actions, size=state_dim) for i in range(episode_length)]
reward_raw_sq = [i+1 for i in range(episode_length)]
next_state_raw_sq = [(i+1) * np.ones(state_dim) for i in range(episode_length)]

print(state_raw_sq)
print(action_raw_sq)
print(reward_raw_sq)
print(next_state_raw_sq)

raw_sq = RawSequence()
for state, action, reward, next_state in zip(state_raw_sq, action_raw_sq, reward_raw_sq, next_state_raw_sq):
    raw_sq.push(state, action, reward, next_state)

# print(raw_sq.state_sq)

episode = EpisodeRecoder(raw_sq, state_value_max=3.0, reward_normalization=10.0, n_local_actions=3)

print('Processed observation sequence\n', episode.observation_sq)
print('Processed global state sequence\n', episode.global_state_sq)
print('Processed next global state sequence\n', episode.next_global_state_sq)
print('Processed reward sequence\n', episode.reward_sq)


