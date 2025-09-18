import numpy as np
from ariel.simulation.environments import *
from A2.Controller import RandomController, NNController

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

ControllerType = RandomController | NNController

CONTROLLER_MAP: dict[str, type[ControllerType]] = {
    "RandomController": RandomController,
    "NNController": NNController,
}

History = np.ndarray