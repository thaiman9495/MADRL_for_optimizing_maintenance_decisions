import numpy as np

from itertools import product


def create_action_space(n_components, n_c_actions) -> np.ndarray:
    """
    This function aims at creating action space at system level

    Args:
        n_components (int):  number of components
        n_c_actions (int): number of actions at component lelvel

    Returns: system action space

    """

    return np.array(tuple(product(range(n_c_actions), repeat=n_components)), dtype=int)


def create_state_space(n_components, n_c_states) -> np.ndarray:
    """
    This function aims to create state spaces for multi-component systems

    Args:
        n_components (int): number of components
        n_c_states (int): number of states in each component

    Returns: state space

    """

    return np.array(tuple(product(range(n_c_states), repeat=n_components)), dtype=int)


def create_state_to_id(n_components: int, n_c_states: int) -> np.ndarray:
    """
    An array used to map from system state to corresponding index

    Args:
        n_components: number of components
        n_c_states: number of component degradation states

    Returns: a mapping from system state to corresponding index

    """
    state_space = create_state_space(n_components, n_c_states)
    state_to_id = np.zeros(tuple(n_c_states for _ in range(n_components)), dtype=int)

    for idx, state in enumerate(state_space):
        state_to_id[tuple(state)] = idx

    return state_to_id


def create_action_to_id(n_components: int, n_c_actions: int) -> np.ndarray:
    """
    An array used to map from system state to corresponding index

    Args:
        n_components: number of components
        n_c_actions: number of component actions

    Returns: a mapping from system state to corresponding index

    """
    action_space = create_action_space(n_components, n_c_actions)
    action_to_id = np.zeros(tuple(n_c_actions for _ in range(n_components)), dtype=int)

    for idx, action in enumerate(action_space):
        action_to_id[tuple(action)] = idx

    return action_to_id
