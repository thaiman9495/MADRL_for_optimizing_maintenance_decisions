import numpy as np
from thai.utility.stochastic_dependence import update_transition_matrix


class Component:
    def __init__(self, transition_matrix: np.ndarray, type_c: int):
        self.P_0 = transition_matrix.copy()              # Inherent transition matrix
        self.P = transition_matrix.copy()                # Transition matrix afected by other componets
        self.n_states = transition_matrix.shape[0]       # Number of states
        self.state = 0                                   # Initialize component's state as new
        self.state_space = np.arange(self.n_states)      # State space
        self.type_c = type_c                             # Component type

    def degrade(self):
        """This function describes component degradtion process"""
        self.state = np.random.choice(self.state_space, p=self.P[self.state, :])

    def reset(self):
        """This function reset the state of component to as good as new (state number zero)"""
        self.state = np.random.randint(self.n_states)

    def update_transition_matrix(self, tau):
        """
        This function aims at update component's transition matrix

        Args:
            tau: degradation interaction factor

        """
        self.P = update_transition_matrix(self.P_0, self.n_states, tau)

    def perform_action(self, intervetion_gain):
        """
        This function aims to calculate component's state after maintenance action

        Args:
            intervetion_gain (int): maintenance action's corresonding intervention gain
        """
        self.state -= intervetion_gain


if __name__ == '__main__':
    P = np.array([[0.60, 0.30, 0.10, 0.00, 0.00],
                  [0.00, 0.50, 0.35, 0.10, 0.05],
                  [0.00, 0.00, 0.50, 0.35, 0.15],
                  [0.00, 0.00, 0.00, 0.40, 0.60],
                  [0.00, 0.00, 0.00, 0.00, 1.00]])

    my_component = Component(P, 1)
    my_component.update_transition_matrix(0.5)
    print(my_component.P_0)
    print(my_component.P)
