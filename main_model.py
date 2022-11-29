import mlflow
import sys

from util import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wind_direction_transformer import WindDirectionTransformer
from sklearn.svm import SVR


class SVRWindPowerPrediction(mlflow.pyfunc.PythonModel):
    def __init__(self, C, epsilon, gamma):

        self.pipeline = Pipeline([
            ("wind_direction_conversion", WindDirectionTransformer()),
            ('min_max_scaling', StandardScaler()),
            ('svr', SVR(C = C, epsilon = epsilon, gamma = gamma))
        ])

    def fit(self,X,y):
        self.pipeline.fit(X, y)
        return self

    def predict(self,context, X):
        return self.pipeline.predict(X)

if __name__ == "__main__":

        best_C = 1.9
        best_eps = 0.4
        best_gamma = "auto"

        model = SVRWindPowerPrediction(best_C, best_eps, best_gamma)

        save_model("WIND_SVR_model", model)

        
                        

        
