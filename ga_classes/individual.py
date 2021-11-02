import numpy as np

from ga_classes.envs import EnvGA


class IndividualP:
    def __init__(self, scenario_count: int, executor_count: int):
        self._s_count = scenario_count
        self._e_count = executor_count
        self._genes = self.generate_genes()
        self._fitness = None

    @property
    def s_count(self):
        return self._s_count

    @property
    def e_count(self):
        return self._e_count

    @property
    def genes(self):
        return self._genes

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, env: EnvGA) -> float:
        raise NotImplemented()

    def generate_genes(self):
        raise NotImplemented()
