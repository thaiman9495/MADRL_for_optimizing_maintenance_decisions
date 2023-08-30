import torch
import yaml

from pathlib import Path
from thai.env.system import System
from config.env.system_structure import structure_function
from thai.agent.dueling_ddqn import DuelingDdqn


# Path config
path_params_agent = 'config/algorithm/dueling_ddqn.yaml'
path_params_env = 'config/env/env.yaml'
path_parent = Path.cwd()

# Load params for configuring agents and environment
with open(path_parent.joinpath(path_params_agent)) as file:
    params_agent = yaml.full_load(file)

with open(path_parent.joinpath(path_params_env)) as file:
    params_env = yaml.full_load(file)

# Initialize interactive environment
env = System(structure_function, **params_env)

# Intialize agent
device = torch.device(params_agent['gpu'] if torch.cuda.is_available() else 'cpu')
agent = DuelingDdqn(env.n_components, env.n_c_actions, env.n_c_states, device, **params_agent)

# Do it
mode = params_agent['run_mode']
path_model = Path(params_agent['path_model'])
path_log = path_parent.joinpath(params_agent['path_log'])

if mode == 1:
    agent.train(env, path_model, path_log)
else:
    agent.evaluate(env, path_model, path_log)



