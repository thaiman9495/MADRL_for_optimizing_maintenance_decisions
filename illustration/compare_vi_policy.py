from torch import load
from pathlib import Path


path_parent = Path.cwd().parent
path_vi_depend = path_parent.joinpath('data/vi')
path_vi_no_depend = path_parent.joinpath('data/vi/no_stochastic')

cost_depend = load(path_vi_depend.joinpath('cost_rate.pt'))
cost_no_depend = load(path_vi_no_depend.joinpath('cost_rate.pt'))

policy_depend = load(path_vi_depend.joinpath('policy.pt'))
policy_no_depend = load(path_vi_no_depend.joinpath('policy.pt'))

for p_depend, p_no_depend in zip(policy_depend.items(), policy_no_depend.items()):
    state, action_depend = p_depend
    _, action_no_depend = p_no_depend
    print(f'{list(state)} & {list(action_depend)} & {list(action_no_depend)}')

print(cost_depend)
print(cost_no_depend)
