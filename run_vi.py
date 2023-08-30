import yaml

from pathlib import Path
from thai.agent.vi import VI
from thai.env.system import System
from config.env.system_structure import structure_function

# Path config
path_params_agent = 'config/algorithm/vi.yaml'
path_params_env = 'config/env/env.yaml'
path_parent = Path.cwd()

# Load params for configuring agents and environment
with open(path_parent.joinpath(path_params_agent)) as file:
    params_agent = yaml.full_load(file)

with open(path_parent.joinpath(path_params_env)) as file:
    params_env = yaml.full_load(file)

# Initialize interactive environment
env = System(structure_function, **params_env)

# Initilize learning agents
path_log = path_parent.joinpath(params_agent['path_log'])
agent = VI(env,  **params_agent)

# Do it
mode = params_agent['run_mode']
if mode == 1:
    agent.train(path_log)
else:
    agent.evaluate(env, path_log)
