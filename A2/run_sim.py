from typing import Literal
import mujoco
from mujoco import viewer
from tqdm import tqdm

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from ctm_types import (
    WorldType, WORLD_MAP,
    ControllerType, CONTROLLER_MAP,
)

def init_sim(world: WorldType):
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    gecko_core = gecko()     # DO NOT CHANGE
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    return model, data, to_track

def main( 
    world_name: str = "SimpleFlatWorld",
    controller_name: str = "RandomController",
    visualise: bool = False,
    record_video: bool = False,
    simulation_steps: int = 10_000,
):
    
    assert world_name in WORLD_MAP, "Invalid world name, choose from: " + ", ".join(WORLD_MAP.keys())
    assert controller_name in CONTROLLER_MAP, "Invalid controller name, choose from: " + ", ".join(CONTROLLER_MAP.keys())

    world = WORLD_MAP[world_name]()
    controller = CONTROLLER_MAP[controller_name]()

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
        for _ in tqdm(range(simulation_steps)):
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