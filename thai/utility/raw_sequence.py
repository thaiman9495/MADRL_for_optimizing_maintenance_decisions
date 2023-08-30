from dataclasses import dataclass


@dataclass
class RawSequence:
    def __post_init__(self):
        self.state_sq = []
        self.reward_sq = []
        self.action_sq = []
        self.next_state_sq = []

    def push(self, state, action, reward, next_state):
        self.state_sq.append(state)
        self.action_sq.append(action)
        self.reward_sq.append(reward)
        self.next_state_sq.append(next_state)

    def get_data(self):
        return self.state_sq, self.action_sq, self.reward_sq, self.next_state_sq

    def __len__(self):
        return len(self.state_sq)
