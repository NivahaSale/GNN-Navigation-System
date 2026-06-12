import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)


def get_congestion(temp, rain, snow, clouds, hour, day, holiday):

    features = np.array([[
        holiday,
        temp,
        rain,
        snow,
        clouds,
        hour,
        day
    ]])

    volume = model.predict(features)[0]

    congestion = min(1.0, volume / 6000)

    return congestion