import torch
import numpy as np

from thai.utility.episode_recorder import EpisodeRecoder


class EpisodeBuffer:
    def __init__(self, capacity, batch_size, episode_len):
        """

        Args:
            capacity (int): maximal number of episodes
            batch_size (int): number of episode in one sampled mini-batch from this buffer
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.episode_len = episode_len

        self.data = np.empty(shape=(self.capacity,), dtype=EpisodeRecoder)

        # Index of the buffer
        self.next_index = 0

        # Number of the non-empty elements in the buffer
        self.size = 0

    def push(self, episode: EpisodeRecoder):
        # Put data
        index = self.next_index
        self.data[index] = episode

        # Update next_index
        self.next_index = (index + 1) % self.capacity

        # Update size of the buffer
        self.size = min(self.capacity, self.size + 1)

    def sample(self, device):
        # Sample indexes unifromally
        batch_index = np.random.randint(self.size, size=self.batch_size)

        # Create a list containing sampled episodes
        batch_episode = [self.data[i] for i in batch_index]

        # Create raw state, state, action, reward, next_state batch
        batch_global_state_sq, batch_next_global_state_sq = [], []
        batch_observation_sq, batch_action_sq, batch_next_observation_sq = [], [], []
        for k in range(self.episode_len):
            batch_global_state_sq.append([ep.global_state_sq[k] for ep in batch_episode])
            batch_next_global_state_sq.append([ep.next_global_state_sq[k] for ep in batch_episode])

            batch_observation_sq.append(np.stack([ep.observation_sq[k] for ep in batch_episode], axis=1))
            batch_action_sq.append(np.stack([ep.action_sq[k] for ep in batch_episode], axis=1))
            batch_next_observation_sq.append(np.stack([ep.next_observation_sq[k] for ep in batch_episode], axis=1))

        batch_reward_sq = np.stack([ep.reward_sq for ep in batch_episode], axis=0)

        # Put these batches into training device
        batch_global_state_sq = torch.FloatTensor(np.array(batch_global_state_sq)).to(device)
        batch_next_global_state_sq = torch.FloatTensor(np.array(batch_next_global_state_sq)).to(device)

        batch_observation_sq = torch.FloatTensor(np.array(batch_observation_sq)).to(device)
        batch_action_sq = torch.LongTensor(np.array(batch_action_sq)).to(device)
        batch_reward_sq = torch.FloatTensor(batch_reward_sq).to(device)
        batch_next_observation_sq = torch.FloatTensor(np.array(batch_next_observation_sq)).to(device)

        return batch_observation_sq, batch_action_sq, batch_reward_sq, batch_next_observation_sq,\
               batch_global_state_sq, batch_next_global_state_sq

    def __len__(self):
        return self.size









