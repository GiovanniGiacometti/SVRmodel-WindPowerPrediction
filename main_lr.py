import mlflow
import sys

from util import *
from sklearn.pipeline import Pipeline
from wind_direction_transformer import WindDirectionTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression



mlflow.set_experiment("giog - polylr+fold")

if __name__ == "__main__":
    with mlflow.start_run():

        X,y = get_dataframe()

        number_of_splits = 5
        poly_features = int(sys.argv[1]) if len(sys.argv) > 1 else 3


        model = Pipeline([
            ("wind_direction_conversion", WindDirectionTransformer()),
            ("polinomial_expansion", PolynomialFeatures(poly_features)),
            ('min_max_scaling', MinMaxScaler()),
            ('lr', LinearRegression())
        ])
        
        for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
            X_train = X.iloc[train].copy()
            y_train = y.iloc[train]
            X_test = X.iloc[test].copy()
            y_test = y.iloc[test]
            model = model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            evaluate(y_train, y_test, model.predict(X.iloc[train].copy()), predictions)
            
        log(["poly_features", "number_of_splits"],[poly_features, number_of_splits])
        
