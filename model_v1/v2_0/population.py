from ga_classes.envs import EnvGA
from ga_classes.population import PopulationP


class Population(PopulationP):
    def __init__(self, individual_count: int, env: EnvGA, individual_class: type,
                 crossover_individuals_count: int = None):
        super().__init__(individual_count, env, individual_class)
        self.ind_class = individual_class
        self.crossover_count = min(crossover_individuals_count, individual_count) \
            if crossover_individuals_count is not None else min(3, individual_count)

    @property
    def individuals(self):
        return self._inds

    @individuals.setter
    def individuals(self, new_individuals):
        if self._validate_individuals(new_individuals):
            self._inds = new_individuals
        else:
            raise Exception("I didn't like that")

    def _get_best_ids(self):
        best_ids = [self._inds.index(x) for x in sorted(self._inds, key=lambda ind: ind.fitness)[:self.crossover_count]]
        return best_ids

    def _validate_individuals(self, individuals) -> bool:
        b = len(self._inds) == len(individuals)
        for i in individuals:
            b = isinstance(i, self.ind_class)
        return b
