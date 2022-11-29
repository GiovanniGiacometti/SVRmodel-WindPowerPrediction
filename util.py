import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def rmse(y1,y2):
    return mean_squared_error(y1,y2,squared=False)

metrics_train = [
        ("RMSE_train",rmse, []),
        ("R2_train", r2_score, []),
        ]

metrics_test = [
    ("RMSE_test", rmse, []),
    ("R2_test", r2_score, []),
    ]

def get_dataframe():
    complete_df = pd.read_json("dataset.json", orient="split")
    speed_df = complete_df[~complete_df["Speed"].isnull()]
    df = speed_df[~speed_df["Direction"].isnull()]
    X = df[["Speed","Direction"]]
    y = df["Total"]
    return X,y

def set_mlflow(name):
    mlflow.set_experiment(name)    

def save_model(name, model):
    mlflow.pyfunc.save_model(name, python_model=model, conda_env = "conda.yaml")


def log(names_list, params_list):
    for name, _, scores in metrics_train+metrics_test:
        print("mean_" + name, np.mean(scores))
        mlflow.log_metric("mean"+name, np.mean(scores))
        mlflow.log_metric("std"+name, np.std(scores))
    for n,p in zip(names_list, params_list):
        print(n, str(p))
        mlflow.log_param(n, str(p))

def evaluate(y_train, y_test, pred_train, pred_test):
    for _, metric, scores in metrics_train:
        score = metric(y_train, pred_train)
        scores.append(score)
    for _, metric, scores in metrics_test:
        score = metric(y_test, pred_test)
        scores.append(score)

