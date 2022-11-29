import mlflow
import sys

import numpy as np
from util import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wind_direction_transformer import WindDirectionTransformer
from sklearn.svm import SVR


mlflow.set_tracking_uri("http://training.itu.dk:5000/")
set_mlflow("giog - svr")

if __name__ == "__main__":
   
        X,y = get_dataframe()

        number_of_splits = 5

        c_range = np.arange(0.5,2,0.1)
        eps_range = np.arange(0.1,0.5,0.1)
        gamma_range = ["auto", "scale"]

        for c in c_range:
            for eps in eps_range:
                for gamma in gamma_range:
                    with mlflow.start_run():
                        print("-------------------")
                        model = Pipeline([
                            ("wind_direction_conversion", WindDirectionTransformer()),
                            ('min_max_scaling', StandardScaler()),
                            ('svr', SVR(C = c, epsilon=eps))
                        ])

                        
                        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
                            X_train = X.iloc[train].copy()
                            y_train = y.iloc[train]
                            X_test = X.iloc[test].copy()
                            y_test = y.iloc[test]
                            model = model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            evaluate(y_train, y_test, model.predict( X.iloc[train].copy()), predictions)

                        log(["C", "epsilon","gamma"],[c, eps,gamma])
                        save_model(model=model)
                

        
