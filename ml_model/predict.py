import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict_congestion(temp, rain, snow, clouds, hour, day, holiday):

    features = np.array([[
        holiday,
        temp,
        rain,
        snow,
        clouds,
        hour,
        day
    ]])

    traffic_volume = model.predict(features)[0]
    congestion = min(1.0, traffic_volume / 6000)

    return congestion


# 🔥 TEST RUN (IMPORTANT)
if __name__ == "__main__":

    result = predict_congestion(
        temp=300,
        rain=0,
        snow=0,
        clouds=40,
        hour=9,
        day=2,
        holiday=0
    )

    print("Predicted congestion:", result)