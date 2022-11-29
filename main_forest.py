import mlflow
import sys

from util import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from wind_direction_transformer import WindDirectionTransformer



set_mlflow("giog - randomforest+fold")

if __name__ == "__main__":
    with mlflow.start_run():

        X,y = get_dataframe()
        
        number_of_splits = 5
        n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    

        model = Pipeline([
            ("wind_direction_conversion", WindDirectionTransformer()),
            ('rf', RandomForestRegressor(n_estimators=n_estimators))
        ])

        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
            X_train = X.iloc[train].copy()
            y_train = y.iloc[train]
            X_test = X.iloc[test].copy()
            y_test = y.iloc[test]
            model = model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            evaluate(y_train, y_test, model.predict(X.iloc[train].copy()), predictions)
        
        log(["n_estimators", "number_of_splits"],[n_estimators, number_of_splits])
        


