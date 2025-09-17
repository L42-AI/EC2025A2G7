from deap import base, creator, tools
import random
import numpy as np
import mujoco
from fitness_functions import evaluate_fitness

class EvolutionManager:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights = (input_size * hidden_size) + (hidden_size * output_size)
        
        # Setup DEAP framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize fitness, cost can be added with -1.0
        creator.create("Individual", list, fitness=creator.FitnessMax) # list is used to store weights in
        
        
