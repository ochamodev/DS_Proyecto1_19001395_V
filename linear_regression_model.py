from dataclasses import dataclass

@dataclass
class LinearRegresionModel:
    yPred: float
    error: float
    iteration: int