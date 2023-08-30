import torch
import numpy as np


class UniformBuffer:
    def __init__(self, capacity, state_size, batch_size, is_action_index=True):
        self.capacity = capacity
        self.state_size = state_size
        self.batch_size = batch_size

        # Create a dictionary to store all data
        self.action_size = 1 if is_action_index else state_size

        self.data = {
                'state': np.zeros(shape=(capacity, state_size), dtype=float),
                'action': np.zeros(shape=(capacity, self.action_size), dtype=int),
                'reward': np.zeros(shape=(capacity, 1), dtype=float),
                'next_state': np.zeros(shape=(capacity, state_size), dtype=float)
            }

        self.batch_state = np.zeros(shape=(batch_size, state_size), dtype=float)
        self.batch_action = np.zeros(shape=(batch_size, self.action_size), dtype=int)
        self.batch_reward = np.zeros(shape=(batch_size, 1), dtype=float)
        self.batch_next_state = np.zeros(shape=(batch_size, state_size), dtype=float)

        # Index of the buffer
        self.next_index = 0

        # Number of the non-empty elements in the buffer
        self.size = 0

    def push(self, state, action, reward, next_state):
        index = self.next_index

        self.data['state'][index] = state
        self.data['action'][index] = action
        self.data['reward'][index] = reward
        self.data['next_state'][index] = next_state

        # Update next_index
        self.next_index = (index + 1) % self.capacity

        # Update size of the buffer
        self.size = min(self.capacity, self.size + 1)

    def sample(self, device):
        batch_index = np.random.randint(self.size, size=self.batch_size)

        for i in range(self.batch_size):
            index = batch_index[i]
            self.batch_state[i] = self.data['state'][index]
            self.batch_action[i] = self.data['action'][index]
            self.batch_reward[i] = self.data['reward'][index]
            self.batch_next_state[i] = self.data['next_state'][index]

        # Put these batches into GPU
        state = torch.FloatTensor(self.batch_state).to(device)
        action = torch.LongTensor(self.batch_action).to(device)
        reward = torch.FloatTensor(self.batch_reward).to(device)
        next_state = torch.FloatTensor(self.batch_next_state).to(device)

        return state, action, reward, next_state

    def __len__(self):
        return self.size
