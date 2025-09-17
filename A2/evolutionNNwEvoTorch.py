# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import evotorch as et

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments import (
    SimpleFlatWorld,
    AmphitheatreTerrainWorld,
    BoxyRugged,
    CraterTerrainWorld,
    PyramidWorld,
    RuggedTerrainWorld,
    TiltedFlatWorld,
)

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history
HISTORY = []


def controller(
    model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track
) -> None:
    """Neural network controller met persistente gewichten, alles in één functie."""

    # THIS SHOULD BE REPLACED BY EVOTORCH WEIGHTS
    if not hasattr(controller, "W1"):
        input_size = len(data.qpos)
        hidden_size = 8
        output_size = model.nu
        controller.W1 = np.random.randn(input_size, hidden_size) * 0.2
        controller.W2 = np.random.randn(hidden_size, hidden_size) * 0.2
        controller.W3 = np.random.randn(hidden_size, output_size) * 0.2

    # Sigmoid activatie
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Forward pass
    inputs = data.qpos
    layer1 = sigmoid(np.dot(inputs, controller.W1))
    layer2 = sigmoid(np.dot(layer1, controller.W2))
    outputs = sigmoid(np.dot(layer2, controller.W3))

    # Scale outputs naar [-pi/2, pi/2]
    scaled_outputs = (outputs - 0.5) * np.pi

    # Voeg delta toe voor smooth beweging
    delta = 0.05
    data.ctrl += scaled_outputs * delta

    # Clip naar joint limits
    data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())



def show_qpos_history(history: list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()


def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()  # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    ########### Set the control callback function ############
    # This is called every time step to get the next action.
    mujoco.set_mjcb_control(lambda m, d: controller(m, d, to_track))
    # mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )


if __name__ == "__main__":
    main()
