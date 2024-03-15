import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression_model import LinearRegresionModel
from sklearn import linear_model


class LinearRegressionScratch:
    def __init__(self) -> None:
        pass

    def calculate_error(self, y: np.ndarray, yPred: np.ndarray, n: int):
        resultSum = 0
        resultSubstraction = ((y - yPred) ** 2)
        
        resultSubstraction = np.sum(((y - yPred)** 2))
        division = (1/(2*n))
        errorNpSum = division * resultSubstraction
        #print(f"error with np.sum {errorNpSum}")
        #print(f"error with loop {errorSum}")
        return errorNpSum
    
    def calculate_gradient_b1(self, yPred: np.ndarray, y: np.ndarray, x: np.ndarray, n: int):
        substraction = (yPred - y) * x
        resultSum = np.sum(substraction)
        
        resultNp = (1 / n) * resultSum
        return resultNp

    def calculate_gradient_b0(self, yPred: np.ndarray, y: np.ndarray, n: int):
        resultDivision = 1/n
        resultSum = np.sum((yPred - y) * 1 )
        result = resultDivision * resultSum
        return result

    def update_parameter_b1(self, b1: float, alpha: float, gradient_b1: float):
        result = b1 - (alpha * gradient_b1)
        return result

    def update_parameter_b0(self, b0: float, alpha: float, gradient_b0: float):
        result = b0 - (alpha * gradient_b0)
        return result
    
    def fit(self, x, y, epochs: int, imprimir_error_cada: int, learning_rate: float):
        onesArray = np.ones_like(x)
        beta_0, beta_1 = 0.4, 0.3
        betas = np.zeros(2)
        betas[0] = beta_0
        betas[1] = beta_1
        errors = []
        models: list[LinearRegresionModel] = []
        observationArray = np.column_stack((x, onesArray))
        yPred = 0.0
        error = 0.0
        for i in range(0, epochs):
            n = i + 1
            if n % imprimir_error_cada == 0:
                print(f"Error obtained in iteration {i + 1}: {error}")
            if i == 0:
                yPred = np.dot(observationArray, betas)
                error = self.calculate_error(y, yPred, n=yPred.size)
                errors.append(error)
                models.append(LinearRegresionModel(yPred=yPred, iteration=i, betas=betas))
            else:
                gradientB1 = self.calculate_gradient_b1(yPred=yPred, y=y, x=x, n=yPred.size)
                gradientB0 = self.calculate_gradient_b0(yPred=yPred, y=y, n=yPred.size)
                beta_0 = self.update_parameter_b0(b0=beta_0, alpha=learning_rate, gradient_b0=gradientB0)
                beta_1 = self.update_parameter_b1(b1=beta_1, alpha=learning_rate, gradient_b1=gradientB1)
                betas[0] = beta_0
                betas[1] = beta_1
                yPred = np.dot(observationArray, betas)
                error = self.calculate_error(y, yPred, n=yPred.size)
                errors.append(error)
                models.append(LinearRegresionModel(yPred=yPred, iteration=i, betas=betas))
        return models, errors
    
    def plot_trained_model_errors(self, errors: list[float]):
        plt.figure(figsize=(20, 10))
        plt.xticks(np.arange(0, len(errors), step=1))
        plt.plot(errors)
        plt.tight_layout()
        plt.title("Change of error across time")
        plt.show()

    def plot_trained_model(self, models: list[LinearRegresionModel], n: int):
        """
        Parameters
        ----------
        models : list[LinearRegresionModel]
            List of models to plot
        n : int
            The number of iterations you are interested.
            Example:
            n = 3. Plots iterations: 3, 6, 9, 12, 15
        """
        plt.figure(figsize=(20, 10))
        dataToPlot = models[::n]
        plt.xticks(np.arange(0, len(models), step=n))
        plt.plot([x.yPred for x in dataToPlot])
        plt.tight_layout()
        plt.title("Evolution of model training across time")
        plt.show()

    def predict_comparison(self, x, manuallyTrainedModel: LinearRegresionModel, sciKitModel: linear_model.LinearRegression):
        onesArray = np.ones_like(x)
        #betas = np.reshape(manuallyTrainedModel.betas, (2, 1))
        betas = manuallyTrainedModel.betas
        manualPrediction = np.dot(np.column_stack((x, onesArray)), betas)
        print(f"manualPred = {manualPrediction}")
