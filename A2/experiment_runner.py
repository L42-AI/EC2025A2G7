from run_sim import main

from ctm_types import Environment
from evolutionNN import Controller

class ExperimentRunner:
    def run_experiment(
        self,
        world: Environment,
        controller: Controller,
        visualise: bool = False,
        record_video: bool = False,
        simulation_steps: int = 10_000,
    ):
        main(
            world=world,
            controller=controller,
            visualise=visualise,
            record_video=record_video,
            simulation_steps=simulation_steps,
        )
