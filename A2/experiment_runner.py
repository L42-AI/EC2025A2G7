from functools import partial
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import numpy as np
from ariel.simulation.environments import SimpleFlatWorld

from run_sim import main
from ctm_types import WORLD_MAP, CONTROLLER_MAP, WorldType, ControllerType, History
from Controller import RandomController
from fitness_functions import get_best_closeness_to_xyz
from visualize import visualise_furthest_point, show_qpos_history

class ExperimentRunner:
    
    def baseline(
        self,
        n: int,
        controller_name: str = RandomController.__name__,
        world_name: str = SimpleFlatWorld.__name__,
        simulation_steps: int = 10_000,
    ):
        
        assert world_name in WORLD_MAP, "Invalid world name, choose from: " + ", ".join(WORLD_MAP.keys())
        assert controller_name in CONTROLLER_MAP, "Invalid controller name, choose from: " + ", ".join(CONTROLLER_MAP.keys())

        dir = Path(__file__).parent / "results"
        dir.mkdir(parents=True, exist_ok=True)
        furthest_points = []

        fitness_func = partial(get_best_closeness_to_xyz, target=np.array([0.0, -10.0, 0.0]))

        for i in tqdm(range(n)):

            world = WORLD_MAP[world_name]()
            controller = CONTROLLER_MAP[controller_name]()

            history = self._run_experiment(
                world=world,
                controller=controller,
                simulation_steps=simulation_steps,
                visualise=False,
                record_video=False,
            )

            path = dir / f"experiment_{world_name}_{controller_name}_{i + 1}.npy"
            np.save(path, history)
            furthest_points.append(fitness_func(history))
        
        visualise_furthest_point(furthest_points)

    def _run_experiment(
        self,
        controller: Optional[ControllerType] = None,
        world: Optional[WorldType] = None,
        simulation_steps: int = 2_000_000,
        visualise: bool = False,
        record_video: bool = False,
    ) -> History:
        
        if controller is None:
            controller = RandomController()
        
        if world is None:
            world = SimpleFlatWorld()

        return main(
            world=world,
            controller=controller,
            simulation_steps=simulation_steps,
            visualise=visualise,
            record_video=record_video,
        ).get_history()
    
  
