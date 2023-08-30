from torch import load
from pathlib import Path

path_parent = Path.cwd().parent

path_branching_wqmix = path_parent.joinpath('data/branching_wqmix')
path_wqmix = path_parent.joinpath('data/wqmix')

training_time_branching_wqmix = load(path_branching_wqmix.joinpath('training_time.pt'))
training_time_wqmix = load(path_wqmix.joinpath('training_time.pt'))

print(f"F-WQMIX: {training_time_branching_wqmix}")
print(f"P-WQMIX: {training_time_wqmix}")
