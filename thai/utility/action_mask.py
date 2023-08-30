import torch


from thai.utility.state_action_space import create_state_space, create_action_space


def create_action_mask(n_c_states, n_c_actions, q_min=-999999.0):
    action_mask = torch.zeros(size=(n_c_states, n_c_actions))
    action_mask[-1, 1] = q_min
    for i in range(1, n_c_actions):
        action_mask[0, i] = q_min

    return action_mask


def create_action_mask_single_agent(n_components, n_c_states, n_c_actions, q_min=-999999.0):
    state_space = create_state_space(n_components, n_c_states)
    action_space = create_action_space(n_components, n_c_actions)

    n_s_states = len(state_space)
    n_s_actions = len(action_space)

    action_mask = torch.zeros(size=(n_s_states, n_s_actions))

    for state_id, state in enumerate(state_space):
        for action_id, action in enumerate(action_space):
            for s, a in zip(state, action):
                if (s == 0 and a != 0) or (s == (n_c_states - 1) and a == 1):
                    # print(f'{state} --> {action}')
                    action_mask[state_id, action_id] = q_min
                    break
    return action_mask
