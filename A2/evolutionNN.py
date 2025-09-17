from abc import ABC, abstractmethod

import numpy as np
import mujoco

from utils import sigmoid

class Controller(ABC):
    def __init__(self):
        self.history = []

    @abstractmethod
    def move(self, model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track) -> None:
        """Generate movements for the robot's joints.
        
        The mujoco.set_mjcb_control() function will always give 
        model and data as inputs to the function. Even if you don't use them,
        you need to have them as inputs.

        Parameters
        ----------

        model : mujoco.MjModel
            The MuJoCo model of the robot.
        data : mujoco.MjData
            The MuJoCo data of the robot.

        Returns
        -------
        None
            This function modifies the data.ctrl in place.
        """
        pass
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_history(self) -> np.ndarray:
        return np.array(self.history)

    def clear(self) -> None:
        self.history.clear()

class RandomController(Controller):
    def move(self, model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track) -> None:
        """Generate random movements for the robot's joints.
        
        The mujoco.set_mjcb_control() function will always give 
        model and data as inputs to the function. Even if you don't use them,
        you need to have them as inputs.

        Parameters
        ----------

        model : mujoco.MjModel
            The MuJoCo model of the robot.
        data : mujoco.MjData
            The MuJoCo data of the robot.

        Returns
        -------
        None
            This function modifies the data.ctrl in place.
        """

        # Get the number of joints
        num_joints = model.nu # 8
        
        # Hinges take values between -pi/2 and pi/2
        hinge_range = np.pi/2
        rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                    high=hinge_range, # pi/2
                                    size=num_joints) 

        # There are 2 ways to make movements:
        # 1. Set the control values directly (this might result in junky physics)
        # data.ctrl = rand_moves

        # 2. Add to the control values with a delta (this results in smoother physics)
        delta = 0.05
        data.ctrl += rand_moves * delta 

        # Bound the control values to be within the hinge limits.
        # If a value goes outside the bounds it might result in jittery movement.
        data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

        # Save movement to history
        self.history.append(to_track[0].xpos.copy())

        ##############################################
        #
        # Take all the above into consideration when creating your controller
        # The input size, output size, output range
        # Your network might return ranges [-1,1], so you will need to scale it
        # to the expected [-pi/2, pi/2] range.
        # 
        # Or you might not need a delta and use the direct controller outputs
        #
        ##############################################

class NNController(Controller):
    def __init__(self, input_size=15, hidden_size=8, output_size=8, weights:np.ndarray=None):
        super().__init__()
        if weights is not None:
            self.W1 = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
            self.W2 = weights[input_size * hidden_size:input_size * hidden_size + (hidden_size * hidden_size)].reshape(hidden_size, hidden_size)
            self.W3 = weights[input_size * hidden_size + (hidden_size * hidden_size):].reshape(hidden_size, output_size)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * 0.2
            self.W2 = np.random.randn(hidden_size, hidden_size) * 0.2
            self.W3 = np.random.randn(hidden_size, output_size) * 0.2

    def move(self, model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track) -> None:
        """Neural network controller with persistent weights, all in one function."""

        # Forward pass
        inputs = data.qpos
        layer1 = sigmoid(np.dot(inputs, self.W1))
        layer2 = sigmoid(np.dot(layer1, self.W2))
        outputs = sigmoid(np.dot(layer2, self.W3))

        # Scale outputs to [-pi/2, pi/2]
        scaled_outputs = (outputs - 0.5) * np.pi

        # Add delta for smooth movement
        delta = 0.05
        data.ctrl += scaled_outputs * delta

        # Clip to joint limits
        data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)

        # Save movement to history
        self.history.append(to_track[0].xpos.copy())

    def clear(self): # TODO: MAKE RESET WEIGHTS
        super().clear()