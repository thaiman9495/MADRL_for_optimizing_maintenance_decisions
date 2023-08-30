import os
import matplotlib.pyplot as plt

from torch import load
from pathlib import Path

path_parent = Path.cwd().parent
path_bdq = path_parent.joinpath('data/bdq')
# path_wqmix = path_parent.joinpath('data/wqmix')
path_branching_wqmix = path_parent.joinpath('data/branching_wqmix')

# BDQ
step_bdq = load(path_bdq.joinpath('step.pt'))
cost_rate_bdq = load(path_bdq.joinpath('cost_rate_train.pt'))

# WQMIX
step_branching_wqmix = load(path_branching_wqmix.joinpath('step.pt'))
cost_rate_branching_wqmix = load(path_branching_wqmix.joinpath('cost_rate_train.pt'))

# WQMIX
# step_wqmix = load(path_wqmix.joinpath('step.pt'))
# cost_rate_wqmix = (path_wqmix.joinpath('cost_rate_train.pt'))

plt.plot(step_bdq, cost_rate_bdq, label='BDQ')
plt.plot(step_branching_wqmix, cost_rate_branching_wqmix, label='Branching WQMIX')
plt.legend()
plt.show()

