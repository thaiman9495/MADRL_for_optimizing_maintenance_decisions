import os
import numpy as np
import matplotlib.pyplot as plt

from torch import load
from pathlib import Path


path_parent = str(Path(os.getcwd()).parent).replace('\\', '/')
path_vi = path_parent + '/data/vi/'
path_dueling_ddqn = path_parent + '/data/dueling_ddqn/'
path_vdn = path_parent + '/data/vdn/'
path_bdq = path_parent + '/data/bdq/'
path_qmix = path_parent + '/data/qmix/'
path_wqmix = path_parent + '/data/wqmix/'

# Dueling DDQN
step_dueling_ddqn = load(path_dueling_ddqn + 'step.pt')
cost_rate_dueling_ddqn = load(path_dueling_ddqn + 'cost_rate.pt')

# VDN
# step_vdn = load(path_vdn + 'step.pt')
# cost_rate_vdn = load(path_vdn + 'cost_rate.pt')

# BDQ
step_bdq = load(path_bdq + 'step.pt')
cost_rate_bdq = load(path_bdq + 'cost_rate.pt')

# QMIX
# step_qmix = load(path_qmix + 'step.pt')
# cost_rate_qmix = load(path_qmix + 'cost_rate.pt')

# WQMIX
step_wqmix = load(path_wqmix + 'step.pt')
cost_rate_wqmix = load(path_wqmix + 'cost_rate.pt')

# VI
step_vi = step_dueling_ddqn
cost_rate_vi = (load(path_vi + 'cost_rate.pt') + 1.5) * np.ones(len(step_vi))

# plt.plot(step_dueling_ddqn, cost_rate_dueling_ddqn, label='Dueling DDQN')
# plt.plot(step_vdn, cost_rate_vdn, label='VDN')
# plt.plot(step_bdq, cost_rate_bdq, label='BDQ')
# plt.plot(step_qmix, cost_rate_qmix, label='QMIX')
plt.plot(step_wqmix, cost_rate_wqmix, label='WQMIX')
# plt.plot(step_vi, cost_rate_vi, label='VI (exact)')
plt.legend()
plt.show()

