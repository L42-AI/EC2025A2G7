from pathlib import Path
from typing import Optional
from functools import partial

import mujoco
import numpy as np
from tqdm import tqdm
from mujoco import viewer

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld

from Controller import RandomController
from fitness_functions import get_best_closeness_to_xyz
from visualize import visualise_furthest_point, show_qpos_history
from ctm_types import WORLD_MAP, CONTROLLER_MAP, WorldType, ControllerType, History

def baseline(
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

        history = single(
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

def single(
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

    return _simulation(
        world=world,
        controller=controller,
        simulation_steps=simulation_steps,
        visualise=visualise,
        record_video=record_video,
    ).get_history()

def _simulation( 
    world: WorldType,
    controller: ControllerType,
    visualise: bool = False,
    record_video: bool = False,
    simulation_steps: int = 10_000,
):

    ############################## INITIALISE SIMULATION ##############################
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    ###################################################################################

    mujoco.set_mjcb_control(lambda m,d: controller.move(m, d, to_track)) # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.

    if visualise:
        viewer.launch(
            model=model,  # type: ignore
            data=data,
        )
    else:
        for _ in range(simulation_steps):
            mujoco.mj_step(model, data)
    
    # If you want to record a video of your simulation, you can use the video renderer.
    if record_video:
        # Non-default VideoRecorder options
        PATH_TO_VIDEO_FOLDER = "./__videos__"
        video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

        # Render with video recorder
        video_renderer(
            model,
            data,
            duration=30,
            video_recorder=video_recorder,
        )
    return controller