import numpy as np


def compute_intervention_gain(state, action, action_wrong, n_components):
    """
    This function aims to compute the corresonding intervention gain of a maintenance action

    Args:
        state (np.ndarray): system state before maintenance
        action (np.ndarray): system action
        action_wrong: feasible action indicator
        n_components: number of components

    Returns: corresponding intervention gain vector

    """
    intervention_gain = np.zeros(n_components, dtype=int)
    if not action_wrong:
        for i in range(n_components):
            if action[i] == 0:
                intervention_gain[i] = 0
            else:
                if action[i] == 1:
                    intervention_gain[i] = np.random.randint(state[i] + 1)
                else:
                    intervention_gain[i] = state[i]

    return intervention_gain


def check_wrong_action(state, action, n_components, n_c_states):
    is_action_wrong = False

    # Check wrong action for new state
    for i in range(n_components):
        if state[i] == 0 and action[i] != 0:
            is_action_wrong = True
            break

    # Check wrong action for corrective maintenance
    if not is_action_wrong:
        for idx, a in enumerate(action):
            if state[idx] == (n_c_states - 1) and a == 1:
                is_action_wrong = True
                break

    return is_action_wrong
