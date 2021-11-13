from copy import deepcopy
from time import time

import numpy.random

from common.env_factory import get_fake_set, get_real_set
from ga_classes.population import PopulationP as Population
from ga_classes.envs import EnvGA
from model_v1.common.individual import Individual


class Runner:
    def __init__(self, env_class, test_set: int = 1, real: bool = False):
        get_set_func = get_fake_set
        if real:
            get_set_func = get_real_set
        self._env, observation, actions = get_set_func(env_class, test_set)
        self.population = Population(100, self._env, Individual)
        self.population.calculate_fitness()
        self.best1 = None
        self.best2 = None

    def run(self):
        converge = False
        counter = 0
        generation = 1
        conv_value = None
        best_fitness = -1
        while not converge:
            self.selection()

            self.crossover()

            if numpy.random.randint(10) % 7 == 0:
                self.mutation()

            self.add_fittest()

            self.population.calculate_fitness()
            best_fitness = self.population.individuals[
                self.population.get_best_ids()[0]
            ].fitness

            if conv_value:
                if abs(conv_value - best_fitness) < 0.01:
                    counter += 1
                else:
                    print("Generation: " + str(generation) +
                          " -> Fittest = " +
                          str(best_fitness))
                    conv_value = best_fitness
            else:
                print("Generation: " + str(generation) +
                      " -> Fittest = " +
                      str(best_fitness))
                conv_value = best_fitness
                counter = 1
            if counter >= 100:
                converge = self.is_converged()
                counter = 1
            generation += 1
        print("Generation: " + str(generation) +
              " -> Fittest = " +
              str(best_fitness))
        best, _ = self.population.get_best_ids()
        self._env.reset()
        self.population.individuals[best].calculate_fitness(env=self._env)
        self._env.render()

    def selection(self):
        best1_id, best2_id = self.population.get_best_ids()
        self.best1 = deepcopy(self.population.individuals[best1_id])
        self.best2 = deepcopy(self.population.individuals[best2_id])

    def crossover(self):
        cross_point = numpy.random.randint(self.population.genes_count)
        best1_genes = self.best1.genes
        best2_genes = self.best2.genes
        for i in range(cross_point):
            t = best1_genes[i]
            best1_genes[i] = best2_genes[i]
            best2_genes[i] = t
        self.best1.genes = best1_genes
        self.best2.genes = best2_genes

    def mutation(self):
        mut_point = numpy.random.randint(self.population.genes_count)

        t = self.best1.genes
        t[mut_point] = numpy.random.randint(self.population.genes_count)
        self.best1.genes = t

        mut_point = numpy.random.randint(self.population.genes_count)

        t = self.best2.genes
        t[mut_point] = numpy.random.randint(self.population.genes_count)
        self.best2.genes = t

    def add_fittest(self):
        self._env.reset()
        self.best1.calculate_fitness(self._env)
        self._env.reset()
        self.best2.calculate_fitness(self._env)

        best = self.best1 if self.best1.fitness > self.best2.fitness else self.best2

        self.population.individuals[self.population.get_worst_id()] = best

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
