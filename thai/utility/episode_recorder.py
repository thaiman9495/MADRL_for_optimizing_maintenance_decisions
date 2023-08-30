import numpy as np

from collections import namedtuple
from thai.utility.raw_sequence import RawSequence

LocalInfo = namedtuple('LocalInfor', ['observation_sq', 'action_sq', 'reward_sq', 'next_observation_sq'])
GlobalInfo = namedtuple('GlobalInfo', ['global_state_sq', 'next_global_state_sq'])


class EpisodeRecoder:
    def __init__(self, raw_sq: RawSequence, state_value_max: float, reward_normalization: float, n_local_actions: int):
        self.n_steps = len(raw_sq)
        self.state_value_max = state_value_max
        self.reward_normalization = reward_normalization
        self.n_local_actions = n_local_actions

        self.n_agents = len(raw_sq.action_sq[0])
        self.agent_id = np.eye(self.n_agents)                      # Agent id via one-hot encoding
        self.action_encoded = np.eye(self.n_local_actions)         # One-hot encoding

        # print(f'Number of agents: {self.n_agents}')
        # print(f'Agent id (one-hot encodeing):\n {self.agent_id}')
        # print(f'One-hot encoding action:\n {self.action_encoded}')

        local_info, global_info = self.process_data(raw_sq)

        self.observation_sq = local_info.observation_sq
        self.action_sq = local_info.action_sq
        self.reward_sq = local_info.reward_sq
        self.next_observation_sq = local_info.next_observation_sq

        self.global_state_sq = global_info.global_state_sq
        self.next_global_state_sq = global_info.next_global_state_sq

    def process_data(self, raw_sq):
        observation_sq = []
        action_sq = []
        next_observation_sq = []
        global_state_sq = []
        next_global_state_sq = []

        last_action = np.zeros(shape=(self.n_agents, self.n_local_actions))      # Initial last action
        last_action[:, 0] = 1.0

        for state_raw, action_raw, reward_raw, next_state_raw in zip(*raw_sq.get_data()):
            # Normalize raw state and raw next state
            state = (state_raw/self.state_value_max).reshape(-1, 1)
            next_state = (next_state_raw/self.state_value_max).reshape(-1, 1)

            # Encode action using one-hot encoding
            action = np.array([self.action_encoded[a] for a in action_raw])

            observation_sq.append(np.concatenate([state, last_action, self.agent_id], axis=1))
            action_sq.append(np.expand_dims(action_raw, axis=1))
            next_observation_sq.append(np.concatenate([next_state, action, self.agent_id], axis=1))

            global_state_sq.append(np.concatenate([state, last_action], axis=1).reshape(1, -1).squeeze())
            next_global_state_sq.append(np.concatenate([next_state, action], axis=1).reshape(1, -1).squeeze())

            last_action = action.copy()

        local_info = LocalInfo(observation_sq=np.array(observation_sq),
                               action_sq=np.array(action_sq),
                               reward_sq=(np.array(raw_sq.reward_sq) / self.reward_normalization),
                               next_observation_sq=np.array(next_observation_sq))

        global_info = GlobalInfo(global_state_sq=np.array(global_state_sq),
                                 next_global_state_sq=np.array(next_global_state_sq))

        return local_info, global_info
