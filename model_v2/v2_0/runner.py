from copy import deepcopy
from time import time

import numpy.random
import numpy as np
from common.env_factory import get_fake_set, get_real_set
from model_v2.common.env import EnvGAM2 as EnvGA
from model_v1.v2_0.population import Population
from model_v2.common.individual import Individual


class Runner:
    def __init__(self, env_class, test_set: int = 1, real: bool = False):
        get_set_func = get_fake_set
        if real:
            get_set_func = get_real_set
        self._env, observation, actions = get_set_func(env_class, test_set)
        self.population = Population(20, self._env, Individual, crossover_individuals_count=10)
        self.population.calculate_fitness()
        self.best = []
        self.generation = 1

    def run(self):
        converge = False
        counter = 0
        conv_value = None
        best_fitness = -1
        while not converge:
            self.selection()

            self.crossover()

            if numpy.random.randint(10) % 7 == 0:
                self.mutation()

            self.add_fittest()

            self.population.calculate_fitness(self._env)
            best_fitness = self.population.individuals[
                self.population.get_best_ids()[0]
            ].fitness

            if conv_value:
                if abs(conv_value - best_fitness) < 0.01:
                    counter += 1
                else:
                    print("Generation: " + str(self.generation) +
                          " -> Fittest = " +
                          str(best_fitness))
                    conv_value = best_fitness
            else:
                print("Generation: " + str(self.generation) +
                      " -> Fittest = " +
                      str(best_fitness))
                conv_value = best_fitness
                counter = 1
            if counter >= 100:
                converge = self.is_converged()
                counter = 1
            self.generation += 1
        print("Generation: " + str(self.generation) +
              " -> Fittest = " +
              str(best_fitness))
        self.print_results()

    def selection(self):
        best_ids = self.population.get_best_ids()
        self.best = (numpy.array(self.population.individuals)[best_ids]).tolist()

    def crossover(self):
        new_best = []
        for ind1_i in range(len(self.best)):
            for ind2_i in range(ind1_i + 1, len(self.best)):
                crossover_index = numpy.random.randint(self.population.genes_count)
                ind_1 = deepcopy(self.best[ind1_i])
                ind_2 = deepcopy(self.best[ind2_i])
                self._swap_genes(ind_1, ind_2, crossover_index)
                new_best.append(ind_1)
                new_best.append(ind_2)
        self.best = new_best

    def mutation(self):
        new_best = self.best.copy()
        for ind in new_best:
            mut_point = numpy.random.randint(low=0, high=len(self.population.env.scenarios))
            t = ind.genes
            t[mut_point] = numpy.random.randint(low=0, high=len(self.population.env.executors))
            ind.genes = t
        self.best.extend(new_best)

    def add_fittest(self):
        self.best.extend(self.population.individuals)
        for ind in self.best:
            self._env.reset()
            ind.calculate_fitness(self._env)

        self.best.sort(key=lambda ind: ind.fitness)

        self.population.individuals = self.best[:len(self.population.individuals)]

    @staticmethod
    def _swap_genes(first, second, crossover_point):
        first_genes = first.genes
        second_genes = second.genes
        for i in range(crossover_point):
            t = first_genes[i]
            first_genes[i] = second_genes[i]
            second_genes[i] = t
        first.genes = first_genes
        second.genes = second_genes

    def print_results(self):
        best = self.population.get_best_ids()[0]
        best_ind = self.population.individuals[best]
        best_ind.evaluate(self._env)
        id_to_o = {x.id: x for x in best_ind.makespans}
        print("Makespan = " + str(max(best_ind.makespans, key=lambda x: x.get_max_time()).get_max_time()))

        for executor in self.population.env.executors:
            if executor.id in id_to_o.keys():
                print(str(id_to_o[executor.id]))
            else:
                print(str(executor))

    def is_converged(self) -> bool:
        counter = {}
        for ind in self.population.individuals:
            if str(ind.genes) in counter.keys():
                counter[str(ind.genes)] = counter[str(ind.genes)] + 1
            else:
                counter[str(ind.genes)] = 1
        for value in counter.values():
            if value >= len(self.population.individuals) * 0.7:
                return True
        return False


if __name__ == "__main__":
    r = Runner(EnvGA, test_set=20, real=True)
    start = time()
    r.run()
    end = time()
    print("Time = " + str((end - start)))
    exit()
