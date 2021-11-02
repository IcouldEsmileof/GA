from common.common_classes import EnvP


class EnvGA(EnvP):
    def __init__(self, scenarios, executors):
        super(EnvGA, self).__init__(scenarios, executors)
        self.max_action = len(self._executors) * len(self._scenarios)
        self._available_actions = {i: True for i in range(self.max_action)}
        self.worst_time = self._get_worst_time()

    def reset(self):
        self._available_actions = {i: True for i in range(self.max_action)}
        for e in self.executors:
            e.clean()

    def step(self, action):
        if not self._is_action_available(action):
            raise Exception("Individual's filter doesn't work")

        executor_index = int(action) % len(self._executors)
        scenario_index = int(action) // len(self._executors)

        self._executors[executor_index].add_scenario(self._scenarios[scenario_index])
        self._remove_actions_for_scenario(scenario_index)

    def get_current_makespan(self):
        current_makespan = 0.0
        for executor in self._executors:
            if executor.get_max_time() > current_makespan:
                current_makespan = executor.get_max_time()
        return current_makespan

    def _get_worst_time(self) -> float:
        self.worst_time = 0.0
        for scenario in self.scenarios:
            for config in scenario.configurations:
                self.worst_time += config
        return self.worst_time

    def _is_action_available(self, action):
        if action not in self._available_actions.keys():
            return False
        return self._available_actions[action]

    def _remove_actions_for_scenario(self, scenario_index):
        for action_number in self._available_actions.keys():
            if action_number // len(self._executors) == scenario_index:
                self._available_actions[action_number] = False

    def render(self):
        print("Makespan = " + str(self.get_current_makespan()))

        for executor in self.executors:
            print(str(executor))
