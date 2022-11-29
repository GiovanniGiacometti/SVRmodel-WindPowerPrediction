import mlflow
import sys

from util import *
from wind_direction_transformer import WindDirectionTransformer
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

set_mlflow("giog - knn+fold")

if __name__ == "__main__":
    with mlflow.start_run():

    
        X,y = get_dataframe()

        number_of_splits = 5
        n_neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    

        model = Pipeline([
            ("wind_direction_conversion", WindDirectionTransformer()),
            ('min_max_scaling', StandardScaler()),
            ('dtr', KNeighborsRegressor(n_neighbors = n_neighbors))
        ])
        
        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
            X_train = X.iloc[train].copy()
            y_train = y.iloc[train]
            X_test = X.iloc[test].copy()
            y_test = y.iloc[test]
            model = model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            evaluate(y_train, y_test, model.predict(X.iloc[train].copy()), predictions)
        
        log(["n_neighbors", "number_of_splits"],[n_neighbors, number_of_splits])
        

