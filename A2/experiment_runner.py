from pathlib import Path

from ariel.simulation.environments import SimpleFlatWorld

from run_sim import main, init_sim
from ctm_types import WorldType, ControllerType
from evolutionNN import Controller, RandomController

from export import export_results

class ExperimentRunner:

    def run_random(self, n: int = 500):
        self._run_experiment(
            n=n,
            controller_name=RandomController.__name__,
            world_name=SimpleFlatWorld.__name__,
        )
    
    def _run_experiment(
        self,
        n: int,
        controller_name: str,
        world_name: str,
        simulation_steps: int = 2_000_000,
    ):
        
        dir = Path(__file__).parent / "results"
        dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            controller = main(
                world_name=world_name,
                controller_name=controller_name,
                simulation_steps=simulation_steps,
                visualise=False,
                record_video=False,
            )
            path = dir / f"experiment_{world_name}_{controller_name}_{190 + i + 1}.npz"
            export_results(path, controller.get_history())
