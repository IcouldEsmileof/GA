import numpy as np
from ga_classes.envs import EnvGA
from ga_classes.individual import IndividualP
from common.common_classes import EvrExecutor


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
        if len(new_genes) == self.s_count:
            self._genes = new_genes
        else:
            raise Exception("Invalid genes")

    @property
    def fitness(self):
        return self._fitness

    def calculate_fitness(self, env: EnvGA) -> float:
        self._fitness = self._calculate_fitness(env)
        return self._fitness

    def generate_genes(self):
        return np.random.random_integers(0, high=(self.e_count - 1),
                                         size=self.s_count) if self.e_count > 0 else np.zeros(self.s_count)

    @staticmethod
    def _reorder(scenarios: list, executor) -> EvrExecutor:
        while len(scenarios) != 0:
            scenario_id = 0
            max_time = executor.get_max_time_if_scenario_is_added(scenarios[scenario_id])
            for i in range(1, len(scenarios)):
                cur_time = executor.get_max_time_if_scenario_is_added(scenarios[i])
                if max_time < cur_time:
                    scenario_id = i
                    max_time = cur_time
            executor.add_scenario(scenarios[scenario_id])
            scenarios.pop(scenario_id)
        return executor

    def _calculate_fitness(self, env: EnvGA) -> float:
        env.reset()
        self.makespans = []
        executors = env.executors
        scenarios = np.array(env.scenarios)
        disp = {}
        for i in range(len(self.genes)):
            if self.genes[i] in disp.keys():
                disp[self.genes[i]].append(i)
            else:
                disp[self.genes[i]] = [i]
        for key in disp.keys():
            ex = EvrExecutor(executors[int(key)])
            scens = scenarios[disp[key]]
            result = self._reorder(scens.tolist(), ex)
            self.makespans.append(result)
        self.best = max(self.makespans, key=lambda x: x.get_max_time())
        return self.best.get_max_time()
