import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from math import ceil
from utils import get_degree_from_rad


class BaseFlarePredictor:

    def __init__(self, history_window, prediction_points, sampling_period=0.1):

        self.history_window = history_window
        self.prediction_points = prediction_points
        self.sampling_period = sampling_period

        self.x_history = []
        self.y_history = []

        self.model_x = self.create_model()
        self.model_y = self.create_model()

    def create_model(self):
        raise NotImplementedError

    def add_sample(self, yaw, pitch):

        self.x_history.append(float(yaw))
        self.y_history.append(float(pitch))

        if len(self.x_history) > self.history_window:
            self.x_history.pop(0)

        if len(self.y_history) > self.history_window:
            self.y_history.pop(0)

    def predict(self):

        if len(self.x_history) < self.history_window:
            return None

        x_pred = self._predict_axis(
            self.model_x,
            self.x_history
        )

        y_pred = self._predict_axis(
            self.model_y,
            self.y_history
        )

        return x_pred[-1], y_pred[-1]

    def _predict_axis(self, model, history):

        history = np.asarray(history)

        n = len(history)

        X_train = (
            np.arange(1, n + 1).reshape(-1, 1)
            * self.sampling_period
        )

        y_train = history

        model.fit(X_train, y_train)

        X_future = (
            np.arange(
                n + 1,
                n + self.prediction_points + 1
            ).reshape(-1, 1)
            * self.sampling_period
        )

        pred = model.predict(X_future)

        return pred

class FlareLinearRegressionPredictor(BaseFlarePredictor):

    def __init__(self, history_window, prediction_points, sampling_period=0.1):

        super().__init__(history_window, prediction_points, sampling_period)

    def create_model(self):
        return LinearRegression()


class FlareRidgeRegressionPredictor(BaseFlarePredictor):

    def __init__(self, history_window, prediction_points, sampling_period=0.1, alpha=1):

        self.alpha = alpha

        super().__init__(history_window, prediction_points, sampling_period)

    def create_model(self):
        return Ridge(alpha=self.alpha)

#test
if __name__ == "__main__":

    yaw_series = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    pitch_series = [0, 2, 4, 6, 8, 10, 12, 14, 16]

    history_window = ceil(10 / 2)

    print("===== LINEAR REGRESSION =====")

    predictor_lr = FlareLinearRegressionPredictor(
        history_window=history_window,
        prediction_points=10,
        sampling_period=0.1
    )

    for yaw, pitch in zip(yaw_series, pitch_series):
        predictor_lr.add_sample(yaw, pitch)

    lr_pred = predictor_lr.predict()

    print("Yaw prediction:")
    print(lr_pred[0])

    print("Pitch prediction:")
    print(lr_pred[1])



    print("\n===== RIDGE REGRESSION =====")

    predictor_rr = FlareRidgeRegressionPredictor(
        history_window=history_window,
        prediction_points=10,
        sampling_period=0.1,
        alpha=0.75
    )

    for yaw, pitch in zip(yaw_series, pitch_series):
        predictor_rr.add_sample(yaw, pitch)

    rr_pred = predictor_rr.predict()

    print("Yaw prediction:")
    print(rr_pred[0])
    print("Pitch prediction:")
    print(rr_pred[1])
