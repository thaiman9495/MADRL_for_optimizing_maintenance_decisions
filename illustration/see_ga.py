import os
import matplotlib.pyplot as plt

from torch import load
from pathlib import Path

path_parent = Path.cwd().parent

path_ga = path_parent.joinpath('data/ga')

cost_rate = load(path_ga.joinpath('cost_rate.pt'))
training_time = load(path_ga.joinpath('training_time.pt'))


print(cost_rate)
print(training_time)
