import torch
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt

from thai.env.system import System


class GenericAlgorithm:
    def __init__(self, n_components, n_c_states, **params):
        self.n_iterations = params['n_iterations']
        self.population_size = params['population_size']
        self.chromonsome_size = n_components
        self.crossover_prob = params['crossover_prob']
        self.mutation_prob = params['mutation_prob']

        self.n_interventions = params['n_interventions']
        self.n_runs = params['n_runs']

        self.n_c_states = n_c_states
        self.is_stochastic_dependence = params['is_stochastic_dependence']

        # Create population
        self.population = np.random.randint(1, self.n_c_states-1, size=(self.population_size, self.chromonsome_size))
        print(self.population)

    def _compute_fitness(self, env: System, threshold):
        cost_rate = np.zeros(self.n_runs)

        for i in range(self.n_runs):
            total_cost = 0.0
            env.reset()
            for _ in range(self.n_interventions):
                state = env.state
                action = self._choose_action(state, threshold)
                _, cost = env.perform_action(action, self.is_stochastic_dependence)
                total_cost += cost

            cost_rate[i] = total_cost / self.n_interventions

        return np.mean(cost_rate)

    def _choose_action(self, state, threshold):
        n_components = len(state)
        action = np.zeros(n_components, dtype=int)

        for i in range(n_components):
            if state[i] < threshold[i]:
                action[i] = 0
            else:
                if state[i] < self.n_c_states - 1:
                    action[i] = 1
                else:
                    action[i] = 2

        return action

    def _select_parent(self, score: np.ndarray):
        n_parents = int(self.population_size / 2)
        total_score = score.sum()

        temp = score.argsort()
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(score))
        max_temp = np.max(rank)
        rank = np.array([max_temp - rank[i] for i in range(len(rank))])
        rank_total = rank.sum()

        p_score = np.array([rank[i] / rank_total for i in range(self.population_size)])

        father_index = np.random.choice(range(self.population_size), (n_parents,), p=p_score)
        mother_index = np.random.choice(range(self.population_size), (n_parents,), p=p_score)

        return father_index, mother_index

    def _generate_new_population(self, father_index, mother_index):
        new_population = []
        for i in range(len(father_index)):
            father = self.population[father_index[i], :]
            mother = self.population[mother_index[i], :]

            if np.random.rand() < self.crossover_prob:
                crossover_point = np.random.randint(self.chromonsome_size)
                # Crossover
                child_1 = np.concatenate((father[:crossover_point], mother[crossover_point:]))
                child_2 = np.concatenate((mother[:crossover_point], father[crossover_point:]))

                # Muttaion
                for j in range(self.chromonsome_size):
                    if np.random.rand() < self.mutation_prob:
                        child_1[j] = np.random.randint(1, self.n_c_states-1)
                        child_2[j] = np.random.randint(1, self.n_c_states-1)
            else:
                child_1 = father.copy()
                child_2 = mother.copy()

            new_population.append(child_1)
            new_population.append(child_2)

        return np.array(new_population, dtype=int)

    def train(self, env, path_log: pathlib.Path):
        # Initialization
        best_index = 0
        best_value = self._compute_fitness(env, self.population[best_index])
        score = np.zeros(self.population_size)

        # Log
        log_iteration = []
        log_best_value = []
        log_best_chromosome = []

        starting_time = datetime.datetime.now()
        # Main loop
        for i in range(self.n_iterations):
            # Evaluate all chromonsomes in the popolation
            for j in range(self.population_size):
                score[j] = self._compute_fitness(env, self.population[j])

            # Obtain new best chromonsome
            best_index = np.argmin(score)
            best_value = score[best_index]

            print(f'iteration {i}: threshold: {self.population[best_index]} --> cost rate: {best_value: .3f}')

            # Hold values for logging
            log_iteration.append(i)
            log_best_value.append(best_value)
            log_best_chromosome.append(self.population[best_index])

            # Select parents
            father_index, mother_index = self._select_parent(score)

            # Generate new generation
            self.population = self._generate_new_population(father_index, mother_index)

        ending_time = datetime.datetime.now()
        training_time = ending_time - starting_time
        print(f"Training time: {training_time}")

        path_log_ = path_log if self.is_stochastic_dependence else path_log.joinpath('no_stochastic')
        torch.save(log_iteration, path_log_.joinpath('iteration.pt'))
        torch.save(log_best_value, path_log_.joinpath('cost_rate.pt'))
        torch.save(log_best_chromosome, path_log_.joinpath('threshold.pt'))
        torch.save(training_time, path_log_.joinpath('training_time.pt'))

        plt.plot(log_iteration, log_best_value)
        plt.show()
