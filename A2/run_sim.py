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

def rename_robot_spec(spec, agent_id: int, pos_offset=(0,0,0)):
    # Rename bodies recursively
    if spec.worldbody is not None:
        def rename_body(body):
            body.name = f"{body.name}_{agent_id}"
            for child in getattr(body, "body", []):
                rename_body(child)
        rename_body(spec.worldbody)

    # Rename joints
    for joint in spec.joints:
        old_name = joint.name
        joint.name = f"{old_name}_{agent_id}"

    # Rename geoms
    for geom in spec.geoms:
        geom.name = f"{geom.name}_{agent_id}"

    # Rename actuators and update their targets
    for act in spec.actuators:
        old_name = act.name
        act.name = f"{old_name}_{agent_id}"

        # Update the transmission target based on type
        if hasattr(act, "joint") and act.joint is not None:
            act.joint = f"{act.joint}_{agent_id}"
        elif hasattr(act, "tendon") and act.tendon is not None:
            act.tendon = f"{act.tendon}_{agent_id}"
        elif hasattr(act, "geom") and act.geom is not None:
            act.geom = f"{act.geom}_{agent_id}"

    return spec

def run_multi_agent_world(
    num_agents: int = 20,
    controller_name: str = "RandomController",
    simulation_steps: int = 6000,
    visualise: bool = False,
    record_video: bool = False,
    spacing: float = 0.5,
) -> list[History]:

    mujoco.set_mjcb_control(None)  # reset callback

    world = SimpleFlatWorld()

    # Create and rename robot specs
    robot_specs = []
    for i in range(num_agents):
        spec = gecko().spec
        spec = rename_robot_spec(spec, agent_id=i, pos_offset=(i*spacing, 0, 0))
        robot_specs.append(spec)
        world.spawn(spec, spawn_position=(i*spacing, 0, 0))

    # Compile world into a single model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Determine actuator slices per robot
    actuators_per_robot = model.nu // num_agents
    slices = [(i*actuators_per_robot, (i+1)*actuators_per_robot) for i in range(num_agents)]

    # Create controllers
    controller_type = CONTROLLER_MAP[controller_name]
    controllers = [controller_type() for _ in range(num_agents)]

    # Global control callback
    def global_control(model, data):
        for ctrl, (start, end) in zip(controllers, slices):
            ctrl.move(model, data, to_track=None)  # you can pass per-robot geoms if needed

    mujoco.set_mjcb_control(global_control)

    # Headless simulation
    if visualise:
        from mujoco import viewer
        viewer.launch(model=model, data=data)
    else:
        for _ in tqdm(range(simulation_steps)):
            mujoco.mj_step(model, data)

    # Optional video recording
    if record_video:
        PATH_TO_VIDEO_FOLDER = "./__videos__"
        video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)
        video_renderer(model, data, duration=simulation_steps, video_recorder=video_recorder)

    # Return history for all controllers
    return [ctrl.get_history() for ctrl in controllers]