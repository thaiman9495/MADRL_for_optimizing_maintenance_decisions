import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from torch import load
from pathlib import Path


def format_func(value, tick_number):
    N = value / 1000000
    return f'{int(N)}'


def smoth_line(cost_rate_log, len_sq=10):
    memmory = deque(maxlen=len_sq)
    cost_rate_list = []
    for cost_rate in cost_rate_log:
        memmory.append(cost_rate)
        cost_rate_list.append(sum(memmory) / len(memmory))

    return cost_rate_list


def load_data(path_parent: Path, algorithm_name: str):
    path_ = path_parent.joinpath(f'data/{algorithm_name}')
    step = load(path_.joinpath('step.pt'))
    cost_raw = load(path_.joinpath('cost_rate.pt'))
    cost = smoth_line(cost_raw)
    return step, cost_raw, cost


plt.rc('font', size=18)                       # Change font size
plt.rc('font', family='serif')                # Change font
# plt.rc('lines', linewidth=1.5)              # Change line width
plt.rc('text', usetex=True)                   # Use Latex

# 1 --> Draw
# 0 --> Not draw
draw_bdq = 0
draw_dueling_ddqn = 0
draw_branching_wqmix = 1
draw_wqmix = 1
draw_vi = 0

# bdq_color = 'magenta'
bdq_color = 'orange'
dueling_ddqn_color = 'blue'
branching_wqmix_color = 'red'
wqmix_color = 'green'
vi_color = 'black'

line_width = 2.5
line_width_raw = 1.0
alpha_raw = 0.5

path_parent = Path.cwd().parent
step_collection = []

fig, ax = plt.subplots(figsize=(8, 6))
if draw_branching_wqmix:
    step_b_wqmix,  cost_b_wqmix_raw, cost_b_wqmix = load_data(path_parent, 'branching_wqmix')
    ax.plot(step_b_wqmix, cost_b_wqmix, label='Branching WQMIX', color=branching_wqmix_color, linewidth=line_width)
    ax.plot(step_b_wqmix, cost_b_wqmix_raw, color=branching_wqmix_color, linewidth=line_width_raw, alpha=alpha_raw)
    step_collection.append(step_b_wqmix)

if draw_dueling_ddqn:
    step_d_ddqn, cost_d_ddqn_raw, cost_d_ddqn = load_data(path_parent, 'dueling_ddqn')
    ax.plot(step_d_ddqn, cost_d_ddqn, label='Deuling DDQN', color=dueling_ddqn_color, linewidth=line_width)
    ax.plot(step_d_ddqn, cost_d_ddqn_raw, color=dueling_ddqn_color, linewidth=line_width_raw, alpha=alpha_raw)
    step_collection.append(step_d_ddqn)

if draw_bdq:
    step_bdq, cost_bdq_raw, cost_bdq = load_data(path_parent, 'bdq')
    ax.plot(step_bdq, cost_bdq, label='BDQ', color=bdq_color, linewidth=line_width)
    ax.plot(step_bdq, cost_bdq_raw, color=bdq_color, linewidth=line_width_raw, alpha=alpha_raw)
    step_collection.append(step_bdq)

if draw_wqmix:
    step_wqmix, cost_wqmix_raw, cost_wqmix = load_data(path_parent, 'wqmix')
    ax.plot(step_wqmix, cost_wqmix, label='WQMIX', color=wqmix_color, linewidth=line_width)
    ax.plot(step_wqmix, cost_wqmix_raw, color=wqmix_color, linewidth=line_width_raw, alpha=alpha_raw)
    step_collection.append(step_wqmix)

if draw_vi:
    path_vi = path_parent.joinpath('data/vi')
    step_vi = step_collection[np.argmax(np.array([len(step) for step in step_collection]))]
    cost_vi = load(path_vi.joinpath('cost_rate.pt')) * np.ones(len(step_vi))
    ax.plot(step_vi, cost_vi, label='VI', color=vi_color, linewidth=line_width)

plt.legend(loc='upper right')
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax.set_xlabel(r'Step ($\times 10^3$)')
ax.set_ylabel("Cost rate")
plt.show()






