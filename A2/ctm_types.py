import numpy as np
from ariel.simulation.environments import *
from evolutionNN import RandomController, NNController

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
    "AmphitheatreTerrainWorld": AmphitheatreTerrainWorld,
    "BoxyRugged": BoxyRugged,
    "CraterTerrainWorld": CraterTerrainWorld,
    "PyramidWorld": PyramidWorld,
    "RuggedTerrainWorld": RuggedTerrainWorld,
    "TiltedFlatWorld": TiltedFlatWorld,
}

ControllerType = RandomController | NNController

CONTROLLER_MAP: dict[str, type[ControllerType]] = {
    "RandomController": RandomController,
    "NNController": NNController,
}

History = np.ndarray