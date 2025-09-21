import numpy as np
from ariel.simulation.environments import *
from Controller import RandomController, NNController, TorchController
from brain import NNBrain, TorchBrain

WorldType = (
    SimpleFlatWorld |
    AmphitheatreTerrainWorld |
    BoxyRugged |
    CraterTerrainWorld |
    PyramidWorld |
    RuggedTerrainWorld |
    TiltedFlatWorld
)

WORLD_MAP: dict[str, type[WorldType]] = {
    "SimpleFlatWorld": SimpleFlatWorld,
    "RuggedTerrainWorld": RuggedTerrainWorld,
    "TiltedFlatWorld": TiltedFlatWorld,
    "BoxyRugged": BoxyRugged,
    "AmphitheatreTerrainWorld": AmphitheatreTerrainWorld,
    "CraterTerrainWorld": CraterTerrainWorld,
    "PyramidWorld": PyramidWorld,
}

ControllerType = RandomController | NNController | TorchController

CONTROLLER_MAP: dict[str, type[ControllerType]] = {
    "RandomController": RandomController,
    "TorchController": TorchController,
    "NNController": NNController,
}

History = np.ndarray

BrainType = NNBrain | TorchBrain