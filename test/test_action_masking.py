from thai.utility.action_mask import create_action_mask, create_action_mask_single_agent

n_components = 3
n_c_states = 4
n_c_actions = 3
q_min = 1.0

action_mask = create_action_mask_single_agent(n_components, n_c_states, n_c_actions)

print(action_mask)


