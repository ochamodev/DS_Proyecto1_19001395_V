from dataclasses import dataclass

import numpy as np

@dataclass
class LinearRegresionModel:
    yPred: float
    iteration: int
    betas: np.ndarray