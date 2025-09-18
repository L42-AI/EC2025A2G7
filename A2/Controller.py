from abc import ABC, abstractmethod

import numpy as np
import mujoco

class Controller(ABC):
    def __init__(self):
        self.history = []

    @abstractmethod
    def get_moves(output_shape: int) -> np.ndarray:
        pass

    def move(self, model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track) -> None:
        # Get the number of joints
        inputs = data.qpos
        output_shape = model.nu # 8

        moves = self.get_moves(inputs, output_shape)

        data.ctrl += moves * 0.05 # smoother physics
        # data.ctrl = moves # junky physics

        data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

        # Save movement to history
        self.history.append(to_track[0].xpos.copy())

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_history(self) -> np.ndarray:
        return np.array(self.history)

    def clear(self) -> None:
        self.history.clear()

class RandomController(Controller):

    def get_moves(self, inputs: np.ndarray, output_shape: int) -> np.ndarray:
        hinge_range = np.pi/2
        return np.random.uniform(
            low = -hinge_range, # -pi/2
            high = hinge_range, # pi/2
            size = output_shape
        )

class NNController(Controller):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray = None):
        super().__init__()
        if weights is not None:
            self.W1 = weights[:input_size * hidden_size].reshape(input_size, hidden_size)
            self.W2 = weights[input_size * hidden_size:input_size * hidden_size + (hidden_size * hidden_size)].reshape(hidden_size, hidden_size)
            self.W3 = weights[input_size * hidden_size + (hidden_size * hidden_size):input_size * hidden_size + 2 * (hidden_size * hidden_size)].reshape(hidden_size, hidden_size)
            self.W4 = weights[input_size * hidden_size + 2 * (hidden_size * hidden_size):].reshape(hidden_size, output_size)
        else:
            self.W1 = np.random.randn(input_size, hidden_size) * 0.2
            self.W2 = np.random.randn(hidden_size, hidden_size) * 0.2
            self.W3 = np.random.randn(hidden_size, hidden_size) * 0.2
            self.W4 = np.random.randn(hidden_size, output_size) * 0.2

    def get_moves(self, inputs: np.ndarray, output_shape: int) -> np.ndarray:
        layer1 = np.tanh(np.dot(inputs, self.W1))
        layer2 = np.tanh(np.dot(layer1, self.W2))
        layer3 = np.tanh(np.dot(layer2, self.W3))
        outputs = np.tanh(np.dot(layer3, self.W4))
        return outputs * (np.pi / 2) # Scale outputs to [-pi/2, pi/2]