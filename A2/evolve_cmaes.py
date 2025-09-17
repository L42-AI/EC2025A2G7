from evotorch import Problem
import torch
from evotorch.algorithms import CMAES
import numpy as np
# Ariel imports
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

from evolutionNN import NNController 

def __init__( self, input_size: int, hidden_size: int, output_size: int):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = hidden_size 

    self.num_weights = (input_size * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)

    self.simulation_steps = 50_000

    #Lower and upper weight bounds
    self.lb = -1.5 * torch.ones(self.num_weights)
    self.ub = 1.5 * torch.ones(self.num_weights)
    
    #build the problem 
    self.problem = Problem (
        objective_sense = "max", #maximization problem 
        objective = self._objective, 
        solution_length=self.num_weights, 
        bounds = (self.lb, self.ub),
        dtype=torch.float32, 
        device="cpu")

    def _objective():
        return 0

    #transition np.array to NNController for the MuJoCo simulation 
    def build_controller(self, weights: np.ndarray) -> NNController:
        ctrl = NNController(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights=weights)
        return ctrl
    

    def evaluate_individual(self, weights: np.ndarray, world_name: str, simulation_steps:int, fitness_fn: callable[[np.ndarray], float])-> float: 
        controller = build_controller(weights)
        return ...
    
    
        
    
