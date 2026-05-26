import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


class FlarePredictor:
    def __init__(
        self,
        pw,
        prediction_points=5,
        sampling_period=0.1
    ):

        self.pw = pw
        self.prediction_points = prediction_points
        self.sampling_period = sampling_period

        self.model = None

        if self.pw < 1:
            self.model = LinearRegression()
        else:
            self.model = Ridge(alpha=1)

    def predict(self, series):
        """
        Parameters
        ----------
        series : list | np.ndarray
            Série temporal histórica.

        Returns
        -------
        np.ndarray
            Valores previstos.
        """

        series = np.asarray(series)

        n = len(series)

        # eixo temporal
        X_train = (np.arange(n).reshape(-1, 1) * self.sampling_period)

        y_train = series

        # treino
        self.model.fit(X_train, y_train)

        # tempos futuros
        X_future = (np.arange(n, n + self.prediction_points).reshape(-1, 1) * self.sampling_period)

        # predição
        predictions = self.model.predict(X_future)

        return predictions


#test
if __name__ == "__main__":


    #LR
    yaw_series = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    predictor_lr = FlarePredictor(pw=0.5, prediction_points=5, sampling_period=0.1)

    lr_pred = predictor_lr.predict(yaw_series)

    print("LR Prediction:")
    print(lr_pred)

    #RR
    yaw_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predictor_rr = FlarePredictor(pw=2.0, prediction_points=5, sampling_period=0.1)

    rr_pred = predictor_rr.predict(yaw_series)

    print("\nRR Prediction:")
    print(rr_pred)
