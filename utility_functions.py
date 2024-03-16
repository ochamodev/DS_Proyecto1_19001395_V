import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from correlation_plot_args import CorrelationPlotArgs

def calculateRangeForColumn(ds: pd.DataFrame, columnName: str):
    max = ds[columnName]['max']
    min = ds[columnName]['min']
    return max - min


def plot_columns(ds: pd.DataFrame):
    columns = ds.columns
    for i, col in enumerate(columns):
        plt.figure(i)
        sns.histplot(ds[col], kde=True)

def calculateCorrelation(x: pd.Series, y: pd.Series):
    return np.corrcoef(x, y)

def plotScatterPlot(args: CorrelationPlotArgs):
    plt.scatter(args.x, args.y, alpha=0.5)
    plt.title(f"{args.title}. Correlation: {args.correlation:.4f}")
    plt.show()

def printCorrelation(colName: str, correlation: float):
    print(f"Correlation for {colName} = {correlation:.4f}")

def histogramErrors(y: np.ndarray, title: str):
    xTicksLabels = ['ManuallyTrained', 'ScikitModel', 'AvgModel']
    fig, ax = plt.subplots(1, 1)
    ax.plot(xTicksLabels, y)
    ax.set_xticks(xTicksLabels)
    ax.set_xticklabels(xTicksLabels, rotation='vertical', fontsize=18)
    ax.set_title(title)
