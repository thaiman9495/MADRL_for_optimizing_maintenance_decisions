import numpy as np


def compute_tau(state: np.ndarray, n_components: int, n_c_states: int, interaction_matrix: np.ndarray, alpha: float):
    tau = np.zeros(n_components)
    for i in range(n_components):
        tau_temp = 0.0
        muy = interaction_matrix[i, :]
        for j in range(n_components):
            tau_temp += muy[j] * (state[j] / (n_c_states - 1)) ** alpha
        tau[i] = tau_temp

    return tau


def update_transition_matrix(P_0: np.ndarray, n_c_states, tau=0.0):
    """
    This function aims to update component's transtion matrix
    Args:
        P_0: original transition matrix
        n_c_states: number of component's states
        tau: degradation interaction factor

    Returns: Updated transition matrix

    """
    P = P_0.copy()
    for i in range(n_c_states - 1):
        for j in range(i, n_c_states):
            if j == i:
                P[i, j] = P_0[i, j] * (1.0 - tau)
            else:
                if j != n_c_states - 1:
                    delta_p = P_0[i, j] / (1.0 - P_0[i, i]) * P_0[i, i]
                    P[i, j] = P_0[i, j] + tau * delta_p
                else:
                    P[i, j] = 1.0 - np.sum(P[i, :-1])

    return P

