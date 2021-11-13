from ga_classes.envs import EnvGA


class PopulationP:
    def __init__(self, individual_count: int, env: EnvGA, individual_class: type):
        self._inds = []
        self.env = env
        self.genes_count = len(self.env.scenarios)
        self.fit = False
        for _ in range(individual_count):
            self._inds.append(individual_class(self.genes_count, len(self.env.executors)))

    @property
    def individuals(self):
        return self._inds

    def get_best_ids(self):
        if not self.fit:
            raise Exception("Fitness is not calculated")
        return self._get_best_ids()

    def _get_best_ids(self):
        if len(self._inds) < 1:
            raise Exception("No elements")
        if len(self._inds) < 2:
            return 0, 0

        best1_id, best2_id = (0, 1) if self._inds[0].fitness < self._inds[1].fitness else (1, 0)

        for i in range(2, len(self._inds)):
            if self._inds[i].fitness < self._inds[best1_id].fitness:
                best2_id = best1_id
                best1_id = i
            elif self._inds[i].fitness < self._inds[best2_id].fitness:
                best2_id = i

        return best1_id, best2_id

    def get_worst_id(self):
        if not self.fit:
            raise Exception("Fitness is not calculated")
        return self._get_worst_id()

    def _get_worst_id(self):
        worst = max(self._inds, key=lambda ind: ind.fitness)
        return self._inds.index(worst)

    def calculate_fitness(self, env: EnvGA = None):
        if env is None and self.env is None:
            raise Exception("No env")

        fitness_env = self.env
        if env is not None:
            fitness_env = env

        for ind in self._inds:
            fitness_env.reset()
            ind.calculate_fitness(fitness_env)


        self.fit = True
