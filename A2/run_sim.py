import mujoco
from mujoco import viewer
from tqdm import tqdm

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld

from ctm_types import WORLD_MAP, CONTROLLER_MAP, WorldType, ControllerType, History

def main( 
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
        video_recorder = VideoRecorder(
            width=320,
            height=240,
            output_folder=PATH_TO_VIDEO_FOLDER
        )

        # Render with video recorder
        video_renderer(
            model,
            data,
            duration=30,
            video_recorder=video_recorder,
        )
    return controller