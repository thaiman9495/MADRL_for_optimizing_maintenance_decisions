import datetime
import pathlib

import numpy as np

from torch import save, load
from itertools import product
from thai.env.system import System
from thai.utility.action_manipulation import check_wrong_action
from thai.utility.state_action_space import create_state_space, create_action_space, create_state_to_id
from thai.utility.stochastic_dependence import compute_tau, update_transition_matrix


class VI:
    def __init__(self, env: System,  **params) -> None:
        # Get some parameters from the environment
        self.n_components = env.n_components
        self.n_c_states = env.n_c_states
        self.n_c_actions = env.n_c_actions
        self.interaction_matrix = env.interaction_matrix
        self.alpha = env.alpha
        self.reward_function = env.reward_function
        self.is_stochastic_dependence = params['is_stochastic_dependence']

        self.gamma = params['discount_factor']
        self.theta = params['estimation_accuracy']

        # Parameters of evaluation mode
        self.n_runs = params['n_runs']
        self.n_interactions = params['n_interactions']

        # Get component's transition matrices
        self.p_component = np.zeros((self.n_components, self.n_c_states, self.n_c_states))
        for i in range(self.n_components):
            self.p_component[i, :, :] = env.component[i].P_0

        self.action_space = create_action_space(self.n_components, self.n_c_actions)
        self.state_space = create_state_space(self.n_components, self.n_c_states)
        self.state_to_id = create_state_to_id(self.n_components, self.n_c_states)

        self.n_s_actions = len(self.action_space)
        self.n_s_states = len(self.state_space)

        self.intervention_gain_space = self._create_intervention_gain_space()
        self.p_system = self._compute_system_transition_matrix(self.is_stochastic_dependence)

    def _compute_system_transition_matrix(self, is_stochastic_depend: bool) -> np.ndarray:
        # Initialize transition matrix at system level
        p_system = np.zeros((self.n_s_states, self.n_s_states))

        # Compute p_system
        for i in range(self.n_s_states):
            # Prepare
            state = self.state_space[i, :]
            p = self.p_component.copy()

            # Update component's transition matrix
            if is_stochastic_depend:
                tau = compute_tau(state, self.n_components, self.n_c_states, self.interaction_matrix, self.alpha)
                for j in range(self.n_components):
                    p[j, :, :] = update_transition_matrix(self.p_component[j, :, :], self.n_c_states, tau[j])

            # Compute system transition probability
            for j in range(self.n_s_states):
                next_state = self.state_space[j, :]
                probability = 1.0
                for k in range(self.n_components):
                    probability *= p[k, state[k], next_state[k]]

                p_system[i, j] = probability

        return p_system

    def _create_intervention_gain_space(self) -> np.ndarray:
        intervention_gain_space = np.zeros((self.n_s_states, self.n_s_actions), dtype=tuple)
        for id_state, state in enumerate(self.state_space):
            for id_action, action in enumerate(self.action_space):
                # Check action wrong
                action_wrong = check_wrong_action(state, action, self.n_components, self.n_c_states)

                if action_wrong:
                    gain = (np.zeros(self.n_components),)
                else:
                    gain_holder = []
                    for i in range(self.n_components):
                        if action[i] == 0:
                            gain_holder.append((0, ))
                        else:
                            if action[i] == 2:
                                gain_holder.append((state[i],))
                            else:
                                gain_holder.append(tuple(range(state[i] + 1)))

                    gain_holder = tuple(gain_holder)
                    gain = tuple(product(*gain_holder))

                intervention_gain_space[id_state, id_action] = np.array(gain, dtype=int)

        return intervention_gain_space

    def _compute_v_new(self, v, id_state, state, id_action, action):
        # Get intervention gain tuple
        gain_list = self.intervention_gain_space[id_state, id_action]

        # Compute of number of possible intervention gains
        n_gains = len(gain_list)

        # Compute update (V_new)
        v_new = 0.0
        for gain in gain_list:
            # Compute state after maintenance and reward
            reward, state_am = self.reward_function(state, action, gain)

            # Get index of state_am in action_space
            id_state_am = self.state_to_id[tuple(state_am)]

            for id_next_state in range(self.n_s_states):
                probability = self.p_system[id_state_am, id_next_state]
                v_new += probability * (reward + self.gamma * v[id_next_state])

        v_new = v_new / n_gains

        return v_new

    def train(self, path_log: pathlib.Path):
        v = - np.random.randn(self.n_s_states)             # initial value function
        delta = 1.0                                        # estimation accuracy

        print('Compute optimal value function')
        starting_time = datetime.datetime.now()
        while delta > self.theta:
            delta = 0.0
            for id_state, state in enumerate(self.state_space):
                v_old = v[id_state]
                v_max = - float('inf')

                # Iterate over all actions to update value function
                for id_action, action in enumerate(self.action_space):
                    v_new = self._compute_v_new(v, id_state, state, id_action, action)
                    v_max = max(v_max, v_new)

                v[id_state] = v_max
                delta = max(delta, abs(v_old - v_max))

            print(f'delta: {delta: .10f}, theta: {self.theta: .5f}')

        print('Get optimal policy from optimal value function')
        optimal_policy = {}
        for id_state, state in enumerate(self.state_space):
            v_max = - float('inf')
            id_action_max = self.action_space[0, :]

            for id_action, action in enumerate(self.action_space):
                v_new = self._compute_v_new(v, id_state, state, id_action, action)
                if v_new > v_max:
                    v_max = v_new
                    id_action_max = id_action

            # Store optimal action in optimal policy
            optimal_policy[tuple(state)] = self.action_space[id_action_max, :]

        ending_time = datetime.datetime.now()
        training_time = ending_time - starting_time
        print('Finishing training', f'Training time: {training_time}', sep='\n')

        # Save data
        path_log_ = path_log if self.is_stochastic_dependence else path_log.joinpath('no_stochastic')
        save(optimal_policy, path_log_.joinpath('policy.pt'))
        save(training_time, path_log_.joinpath('training_time.pt'))

    def evaluate(self, env: System, path_log: pathlib.Path):
        path_log_ = path_log if self.is_stochastic_dependence else path_log.joinpath('no_stochastic')
        policy = load(path_log_.joinpath('policy.pt'))

        for state, action in policy.items():
            print(f'{state} -----> {action}')

        average_cost = np.zeros(self.n_runs)
        for i in range(self.n_runs):
            total_cost = 0.0
            env.reset()
            for _ in range(self.n_interactions):
                state = env.state
                action = np.array(policy[tuple(state)], dtype=int)
                _, cost = env.perform_action(action, is_stochastic_dependence=True)
                total_cost += cost

            average_cost[i] = total_cost / self.n_interactions

        mean_cost_rate = np.mean(average_cost)
        std_cost_rate = np.std(average_cost)

        # cost_rate_save = mean_cost_rate
        cost_rate_save = mean_cost_rate + std_cost_rate
        save(cost_rate_save, path_log_.joinpath('cost_rate.pt'))

        print(mean_cost_rate)
        print(std_cost_rate)


if __name__ == '__main__':
    n_components = 5
    n_c_states = 4
    is_random = True

