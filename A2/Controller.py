from abc import ABC, abstractmethod

from brain import TorchBrain, NNBrain
import numpy as np
import mujoco
import torch

class Controller(ABC):
    def __init__(self):
        self.history = []

    @abstractmethod
    def get_moves(self, inputs: np.ndarray, output_shape: int) -> np.ndarray:
        pass

    def move(self, model: mujoco._structs.MjModel, data: mujoco._structs.MjData, to_track) -> None:
        qpos = data.qpos # 15
        qvel = data.qvel # 14
        output_shape = model.nu # 8

        q_input = np.concatenate([qpos, qvel]) # 15 + 14 = 29

        moves = self.get_moves(q_input, output_shape)

        data.ctrl += moves * 0.05
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

class TorchController(Controller):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray = None):
        super().__init__()
        self.brain = TorchBrain(input_size, hidden_size, output_size, weights)

    def get_moves(self, inputs: np.ndarray, output_shape: int) -> np.ndarray:
        # We are not using gradients here, so no need to track them
        with torch.no_grad():
            out = self.brain(torch.tensor(inputs, dtype=torch.float32))
        return out
    
class NNController(Controller):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray = None):
        super().__init__()
        self.brain = NNBrain(input_size, hidden_size, output_size, weights)

    def get_moves(self, inputs: np.ndarray, output_shape: int) -> np.ndarray:
        return self.brain.forward(inputs)