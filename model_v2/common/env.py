import numpy as np

from ga_classes.envs import EnvGA


class EnvGAM2(EnvGA):
    def __init__(self, scenarios, executors):
        super(EnvGA, self).__init__(scenarios, executors)
        self._scenarios_def = list(map(lambda scenario: self._calculate_config_sum(scenario), scenarios))
        self._executors_def = list(map(lambda executor: sum(executor.queues), executors))
        self._scheduled_scenarios = 0

    def reset(self):
        self._scheduled_scenarios = 0
        for e in self.executors:
            e.clean()
        self._executors_def = list(map(lambda executor: sum(executor.queues), self.executors))

    def step(self, action):
        self._executors_def[action] += self._scenarios_def[self._scheduled_scenarios]
        self._executors[action].add_scenario(self._scenarios[self._scheduled_scenarios])
        self._scheduled_scenarios += 1


    def get_current_makespan(self):
        current_makespan = 0.0
        for executor in self._executors_def:
            if executor > current_makespan:
                current_makespan = executor
        return current_makespan

    @staticmethod
    def _calculate_worst_time(scenarios: list, executors_count: int) -> float:
        scenarios = list(map(lambda scenario: EnvGAM2._calculate_config_sum(scenario), scenarios))
        scenarios.sort(reverse=True)

        max_sum = sum(scenarios)

        if len(scenarios) >= 2:
            return max(scenarios[0] + scenarios[1], max_sum // executors_count)
        else:
            return max_sum

    def _is_action_available(self, action):
        if action not in self._available_actions.keys():
            return False
        return self._available_actions[action]

    def _remove_actions_for_scenario(self, scenario_index):
        for action_number in self._available_actions.keys():
            if action_number // len(self._executors) == scenario_index:
                self._available_actions[action_number] = False

    def render(self, mode='human'):
        print("Makespan = " + str(self.get_current_makespan()))

        for i in range(len(self._executors_def)):
            print("Executor " + str(i) + " = " + str(self._executors_def[i]))
            print("Scenarios: " + str(self._executors[i]))

    @staticmethod
    def _calculate_config_sum(scenario):
        if isinstance(scenario, int):
            return scenario
        result = 0
        for config in scenario.configurations:
            result += config
        return result