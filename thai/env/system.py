import numpy as np

from thai.env.component import Component
from thai.utility.stochastic_dependence import compute_tau
from thai.utility.action_manipulation import check_wrong_action, compute_intervention_gain


class System:
    def __init__(self, structure_function, **params):
        # Configure system's parameters
        self.n_components = params['n_components']
        self.n_c_states = params['n_c_states']
        self.n_c_actions = params['n_c_actions']
        self.n_types = params['n_types']
        self.interaction_matrix = np.array(params['interaction_matrix'])
        self.alpha = params['alpha']
        self.structure_function = structure_function

        # Config parmeters for computing cost
        self.c_0 = params['system_setup_cost']
        self.c_t = params['component_setup_cost']
        self.c_ins = params['inspection_cost']
        self.c_r = params['replacement_cost']
        self.c_dt = params['downtime_cost']
        self.c_penalty = params['penalty_cost']
        self.eta = params['imperfect_constant']

        # List holds all components
        type_temp = params['types']
        P_type = np.array(params['matrix_type'])
        self.component = [Component(P_type[type_temp[i]], type_c=type_temp[i]) for i in range(self.n_components)]

    @property
    def state(self):
        return np.array([component.state for component in self.component], dtype=int)

    @state.setter
    def state(self, value):
        for component, val in zip(self.component, value):
            component.state = val

    def reset(self):
        """ Reset all components to as good as new (state number zero) """
        for component in self.component:
            component.reset()

    def _degrade(self, is_stochastic_dependence):
        """ This function compute system' state s_{k+1} from state after maintenance s'_k """

        if is_stochastic_dependence:
            # Compute tau
            tau = compute_tau(self.state, self.n_components, self.n_c_states, self.interaction_matrix, self.alpha)

            # Update component transition matrix
            for i in range(self.n_components):
                self.component[i].update_transition_matrix(tau[i])

        for component in self.component:
            component.degrade()

    def _is_system_failed(self, state):
        """
        This function aims at checking wheather or not the system is failed at inspection time

        Args:
            state: system state before maintenance

        Returns: 0 if the system sitll functioning or 1 if the system is in failed state
        """

        # s is the array with size of n_components elements
        # Each element of s expresses component's state corresponding its order
        # If s[i] = 0 ---> Component i is in failed state
        # If s[i] = 1 ---> Component i is still functioning

        s = np.ones((self.n_components,))
        for i in range(self.n_components):
            if state[i] == self.n_c_states - 1:
                s[i] = 0

        # Check whether the system is still functioning
        # s_system = 0 ---> The system is still functioning
        # s_system = 1 ---> The system fails
        s_system = 1.0 - self.structure_function(s)

        return s_system

    def _compute_total_cost(self, state, action, action_wrong, intervention_gain):
        if action_wrong:
            total_cost = self.c_penalty
            return total_cost
        else:
            # Check whether system is failed
            I_dt = self._is_system_failed(state)

            # Compute number of components being maintained
            n_maintained_components = np.where(action > 0, 1, 0).sum()

            # Compute number of maintained components for each types
            n_maintained_components_type = np.zeros(self.n_types, dtype=int)
            for idx, a in enumerate(action):
                if a > 0:
                    n_maintained_components_type[self.component[idx].type_c] += 1

            # Compute individual costs
            individual_cost = np.zeros(self.n_components)
            for i in range(self.n_components):
                type_index = self.component[i].type_c
                if action[i] != 0:
                    cost = self.c_r[type_index] * (intervention_gain[i] / state[i]) ** self.eta[type_index]
                    individual_cost[i] = self.c_ins[type_index] + self.c_0 + self.c_t[type_index] + cost
                else:
                    individual_cost[i] = self.c_ins[type_index]

            # Compute total cost
            save_c_0 = 0.0 if n_maintained_components == 0 else (n_maintained_components - 1) * self.c_0

            save_c_t = 0.0
            for idx, n in enumerate(n_maintained_components_type):
                if n != 0:
                    save_c_t += (n - 1) * self.c_t[idx]

            total_cost = individual_cost.sum() - save_c_0 - save_c_t + I_dt * self.c_dt
            return total_cost

    def perform_action(self, action, is_stochastic_dependence):
        # Check worng action
        is_action_wrong = check_wrong_action(self.state, action, self.n_components,  self.n_c_states)

        # Compute intervention gain
        intervention_gain = compute_intervention_gain(self.state, action, is_action_wrong, self.n_components)

        # Compute cost
        cost = self._compute_total_cost(self.state, action, is_action_wrong, intervention_gain)

        # Implement action
        for i in range(self.n_components):
            self.component[i].perform_action(intervention_gain[i])

        # Let the system degrade gradually
        self._degrade(is_stochastic_dependence)

        return self.state, cost

    def reward_function(self, state, action, intervention_gain):
        # Check worng action
        action_wrong = check_wrong_action(state, action, self.n_components, self.n_c_states)
        cost = self._compute_total_cost(state, action, action_wrong, intervention_gain)
        if action_wrong:
            state_am = state.copy()
        else:
            state_am = state - intervention_gain
        return -cost, state_am


if __name__ == '__main__':
    pass
    # my_system = System()
    # print(my_system.component[0].P_0)
    # state = np.array([0, 1, 2, 0, 4])
    # intervention_gain = np.array([0, 1, 2, 0, 4])
    # my_system.degrade()
