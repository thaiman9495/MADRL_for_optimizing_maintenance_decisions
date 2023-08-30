import torch
import yaml

from pathlib import Path
from thai.env.system import System
from config.env.system_structure import structure_function
from thai.agent.branching_wqmix import BranchingWqmix

# Path config
path_params_agent = 'config/algorithm/branching_wqmix.yaml'
path_params_env = 'config/env/env.yaml'
path_parent = Path.cwd().parent

# Load params for configuring agents and environment
with open(path_parent.joinpath(path_params_agent)) as file:
    params_agent = yaml.full_load(file)

with open(path_parent.joinpath(path_params_env)) as file:
    params_env = yaml.full_load(file)

# Initialize interactive environment
env = System(structure_function, **params_env)

# Intialize agent
device = torch.device(params_agent['gpu'] if torch.cuda.is_available() else 'cpu')
agent = BranchingWqmix(env.n_components, env.n_c_actions, env.n_c_states, device, **params_agent)

path_model = Path(params_agent['path_model']).joinpath('stochastic')
path_log = path_parent.joinpath(params_agent['path_log'])

print(path_model)
log_cost_rate = []

agent.q_tot_net.eval()

# Load policy
step = 1900000
policy = torch.load(path_model.joinpath(f'policy_{step}.pt'))
agent.q_tot_net.load_state_dict(policy)
env.reset()
for _ in range(400):
    state = env.state
    action = agent.choose_action(agent.q_tot_net, state, epsilon=0.0)
    _, cost = env.perform_action(action, is_stochastic_dependence=True)

    print(f'{state} --> {action}')
