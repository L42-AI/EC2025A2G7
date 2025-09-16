from itertools import chain

from ariel.simulation.tasks import tasks, _task_fitness_function_map_

def get_fitness_function(name: str):
    assert name in tasks, f"Task '{name}' not found. Available tasks: {tasks}"
    return _task_fitness_function_map_[name]

def get_all_fitness_functions():
    return list(chain.from_iterable(_task_fitness_function_map_.values()))
