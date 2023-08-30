import torch
import numpy as np

episode_length = 2
state_dim = 5
n_actions = 3
batch_size = 6

state_episode = [i * np.ones(state_dim) for i in range(episode_length)]
action_episode = [np.random.randint(n_actions, size=state_dim) for i in range(episode_length)]
reward_episode = [np.atleast_1d(i+1) for i in range(episode_length)]
next_state_episode = [(i+1) * np.ones(state_dim) for i in range(episode_length)]

# print(state_episode)
# print(action_episode)
# print(reward_episode)
# print(next_state_state_episode)

last_action = np.zeros(state_dim, dtype=int)
state_episode_processed = []
action_episode_processed = []
next_state_episode_processed = []
for i in range(episode_length):
    state_episode_processed.append(np.stack([state_episode[i], last_action], axis=1))
    next_state_episode_processed.append(np.stack([next_state_episode[i], action_episode[i]], axis=1))
    action_episode_processed.append(np.expand_dims(action_episode[i], axis=1))
    last_action = action_episode[i]

state_episode_processed = np.array(state_episode_processed)
action_episode_processed = np.array(action_episode_processed)
next_state_episode_processed = np.array(next_state_episode_processed)
# print('Sequence of processed states\n', state_episode_processed)
print('Sequence of processed actions\n', action_episode_processed)
# print('Sequence of next processed states\n', next_state_episode_processed)

episode = {'state': state_episode_processed,
           'action': action_episode_processed,
           'reward': reward_episode,
           'next_state': next_state_episode_processed}

# print(episode['action'])

batch_episode = [episode.copy() for i in range(batch_size)]

# Note
# np.stack([ep['state'][k] for ep in batch_episode], axis=1): batch_state at timestamp k
# batch_state_episode[k, i]: batch_state of agent i at timestamp k
batch_state_episode = np.array([np.stack([ep['state'][k] for ep in batch_episode], axis=1)
                                for k in range(episode_length)])
batch_action_episode = np.array([np.stack([ep['action'][k] for ep in batch_episode], axis=1)
                                for k in range(episode_length)])
batch_reward_episode = np.array([np.stack([ep['reward'][k] for ep in batch_episode], axis=0)
                                for k in range(episode_length)])
batch_next_state_episode = np.array([np.stack([ep['next_state'][k] for ep in batch_episode], axis=1)
                                     for k in range(episode_length)])


# print(batch_state_episode[0, 0])
print(batch_action_episode[0, 0])
# print(batch_reward_episode)
# print(batch_reward_episode[0])


