import torch
import yaml

from pathlib import Path
from thai.env.system import System
from config.env.system_structure import structure_function
from thai.agent.wqmix import Wqmix
from thai.param.wqmix import WqmixParam

# Path config
path_params_agent = 'config/algorithm/wqmix.yaml'
path_params_env = 'config/env/env.yaml'
path_parent = Path.cwd()

# Initialize interactive environment
with open(path_parent.joinpath(path_params_env)) as file:
    param_env = yaml.full_load(file)

env = System(structure_function, **param_env)

# Intialize agent
with open(path_parent.joinpath(path_params_agent)) as file:
    param_agent = yaml.full_load(file)

param_agent = WqmixParam(**param_agent)
param_agent.n_agents = env.n_components
param_agent.input_max = 3.0
param_agent.n_actions = env.n_c_actions
device = torch.device(param_agent.gpu if torch.cuda.is_available() else 'cpu')

agent = Wqmix(device, param_agent)

# Do it
path_model = Path(param_agent.path_model)
path_log = path_parent.joinpath(param_agent.path_log)

if param_agent.run_mode == 1:
    agent.train(env, path_model, path_log)
else:
    agent.evaluate(env, path_model, path_log)



