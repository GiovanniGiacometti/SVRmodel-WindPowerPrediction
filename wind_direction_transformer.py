from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class WindDirectionTransformer(TransformerMixin, BaseEstimator):

    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        x["Direction"] = self.lab_to_deg(x, "Direction") #convert to degrees
        x["Direction"] = x["Direction"] * np.pi / 180 #convert to radians
        x["WindX"] = np.cos(x["Direction"]) * x["Speed"]
        x["WindY"] = np.sin(x["Direction"]) * x["Speed"]
        x.drop(["Direction", "Speed"], axis = 1, inplace=True)
        return x


    def lab_to_deg(self, ds, feature):
        map_lab_to_deg = {
            'S' : 180, 'SW' : 225, 'E' : 90, 'SE':135, 'SSE' : 157.5, 'SSW' : 202.5, 'WSW' : 247.5, 'W' : 270, 'WNW' : 292.5, 'NW' : 315, 'NNW' : 337.5,
            'N' : 0 , 'NNE' : 22.5 , 'ESE' : 112.5, 'NE' : 45, 'ENE' : 67.5
        }
        return ds[feature].apply(lambda x: map_lab_to_deg[x])