from dataclasses import dataclass
import pandas as pd

@dataclass
class CorrelationPlotArgs:
    #x: pd.Series, y: pd.Series, correlation: float
    x: pd.Series
    y: pd.Series
    correlation: float
    title: str