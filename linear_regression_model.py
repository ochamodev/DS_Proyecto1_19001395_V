from dataclasses import dataclass

@dataclass
class LinearRegresionModel:
    yPred: float
    iteration: int
    betas: list[float]