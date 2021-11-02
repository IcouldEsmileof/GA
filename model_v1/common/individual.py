import sys

import numpy as np
from ga_classes.envs import EnvGA
from ga_classes.individual import IndividualP


class Individual(IndividualP):
    def __init__(self, scenario_count: int, executor_count: int):
        super(Individual, self).__init__(scenario_count, executor_count)

    @property
    def s_count(self):
        return self._s_count

    @property
    def e_count(self):
        return self._e_count

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, new_genes: list):
        self._genes = self._filter(new_genes)

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, env: EnvGA) -> float:
        for gene in self.genes:
            env.step(gene)
        self._fitness = env.get_current_makespan()
        return self._fitness

    def generate_genes(self):
        return self._filter(
            np.random.randint(0, high=self.s_count * self.e_count, size=self.s_count, dtype=np.int64))

    def _filter(self, arr) -> []:
        result = []
        d = {}
        for a in arr:
            executor_index = int(a) % self.e_count
            scenario_index = int(a) // self.e_count
            b = True
            while b:
                if scenario_index in d.keys():
                    scenario_index = (scenario_index + 1) % self.s_count
                else:
                    b = False
                    d[scenario_index] = executor_index
                    result.append(scenario_index * self.e_count + executor_index)
        return result
