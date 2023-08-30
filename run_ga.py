import torch
import yaml

from pathlib import Path
from thai.env.system import System
from config.env.system_structure import structure_function
from thai.agent.ga import GenericAlgorithm

# Path config
path_params_agent = 'config/algorithm/ga.yaml'
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
agent = GenericAlgorithm(n_components=env.n_components, n_c_states=env.n_c_states, **params_agent)

# Do it
path_log = path_parent.joinpath(params_agent['path_log'])
agent.train(env, path_log)




