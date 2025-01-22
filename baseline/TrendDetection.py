import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

from sklearn.metrics import log_loss

class TSTrendDetection():
    def __init__(self, bandwidth = 1.0):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth = self.bandwidth)
        self.scaler = StandardScaler()

    def fit_mean_shift(self, X):
        """
        Mean Shift model fitting.

        Parameters
        ----------
        X: np.ndarray (shape = (samples_count, 2)) :
            first samples timedelta from first sample
            second samples values
        or
        X: pd.DataFrame:
            has column timedelta from first sample
            has column value

        Returns
        -------
        X: np.ndarray (shape = (samples_count, 2)) :
            first samples time
            second samples values
        labels: np.ndarray:
            np.ndarray (shape = (samples_count,))
        clustered_ts_value: list:
            list of np.ndarray of clustered values
        clustered_ts_time: list:
            list of np.ndarray of clustered times
        clustered_ts_cofs: list:
            list of slopes of linear regression
        clustered_ts_intercepts: list:
            list of intercepts of linear regression
        """
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X.time = X.time.dt.total_seconds()
            X = X[['time', 'value']].values
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be np.ndarray or pd.DataFrame.")

        # Scaling ts data
        scaler = self.scaler

        print(X)
        X_scaled = scaler.fit_transform(X)
        print(X_scaled)

        # Clustering ts data
        self.mean_shift.fit(X_scaled)
        labels = self.mean_shift.labels_

        # Linear regressions for clustered data
        clusters = np.unique(labels)

        clustered_ts_values = []
        clustered_ts_times = []
        clustered_ts_cofs = []
        clustered_ts_intercepts = []

        for cluster in clusters:
            clustered_series = X[np.where(labels == cluster)]

            times, values, cof, intercept = self._fit_linear_regression(
                clustered_series)

            clustered_ts_values.append(values)
            clustered_ts_times.append(times)
            clustered_ts_cofs.append(cof[0])
            clustered_ts_intercepts.append(intercept)

        return (X, labels, clustered_ts_times, clustered_ts_values,
                clustered_ts_cofs, clustered_ts_intercepts)

    def _fit_linear_regression(self, clustered_series):
        """fit_linear_regression
        Parameters
        ----------
        clustered_series: np.ndarray (shape = (samples of cluster, 2)) :
            first samples time
            second samples values

        Returns
        -------
        X: np.ndarray (shape = (samples_count, 2)) :
            first samples time
            second samples values
        clustered_ts_value: np.ndarray:
            np.ndarray of clustered values
        clustered_ts_time: np.ndarray:
            np.ndarray of clustered times
        clustered_ts_cofs: float:
            slope of a linear regression
        clustered_ts_intercepts: float:
            intercept of a linear regression
        """
        model = LinearRegression()
        model.fit(clustered_series[:, 0].reshape(-1, 1),
                  clustered_series[:, 1].reshape(-1, 1)
                  )

        clustered_ts_value = model.predict(
            clustered_series[:, 0].reshape(-1, 1)
            )
        clustered_ts_time = clustered_series[:, 0]

        # a - slope of a linear regression
        clustered_ts_cofs = model.coef_[0]

        # b - intercept of a linear regression
        clustered_ts_intercepts = model.intercept_

        return (clustered_ts_time, clustered_ts_value,
                clustered_ts_cofs, clustered_ts_intercepts)

    def predict(self, context, model_input, params=None):
        """
        Prediction method for the custom model.

        Parameters:
        -----------
        context : Any
            Ignored in this example. It's a placeholder for additional data or utility methods.

        model_input : tuple
            The input DataFrame or ndarray classified as normal or has anomaly slope.
            and alpha threshold.

        params : dict, optional
            Additional prediction parameters. Ignored.

        Returns:
        --------
        bool
            Has anomaly TS or not.
        """
        return self._predict_internal(model_input[0], model_input[1])

    def _predict_internal(self, X, alpha):
        """
        Prediction using slopes of linear regressions.
        if true then anomaly detected in time series
        """
        _, _, _, _, clustered_ts_cofs, _ = self.fit_mean_shift(X)

        return (np.array(clustered_ts_cofs) > alpha).astype(int)

    def downsample(self, X, smoothing_window = None, skip_window = None):
        """
        Method for time series downsampling.

        Parameters:
        -----------
        X : pd.DataFrame
            first column timedelta
            second column float values
        Returns:
        --------
        pd.DataFrame
            downsampled X.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Time series must be a pd.Series.")
        X = X.copy()
        if smoothing_window != None:
            X.value = X.value.rolling(window=smoothing_window,
                                                    min_periods=1).median()

        if (skip_window != None):
            X.index = X.time
            X = X.value.resample(skip_window).median().fillna(0)
            X = pd.DataFrame({'time': X.index, 'value': X.values})
        return X

    def _binary_cross_entropy(self, threshold, slopes, labels):
        """
        binary cross entropy
        ----------
        threshold: float:
            finding best threshold, by optimizing bce
        slopes: list, array:
            list of slopes, used as thresholds
        labels: np.ndarray (shape = (samples of cluster, 2)) :
            labeled anomalies: 1 if anomaly, 0 otherwise
        Returns
        -------
        best_threshold: float:
            threshold corresponding to highest f1 score
        """
        predictions = (slopes > threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return log_loss(labels, predictions)

    def evaluate_thresholds(self, slopes, labels):
        """
        A method for finding best threshold using binary cross entropy
        ----------
        slopes: list, array:
            list of slopes of linear regressions,
        labels: np.ndarray (shape = (samples of cluster, 2)) :
            labeled anomalies: 1 if anomaly, 0 otherwise
        Returns
        -------
        best_threshold: float:
            threshold corresponding to best cross entropy result
        """
        if len(slopes) < 2:
          return slopes[0]

        initial_threshold = np.mean(slopes)

        # binary cross entropy minimization
        result = minimize(self._binary_cross_entropy, np.mean(slopes),
                          args=(slopes, labels),
                          bounds=[(min(slopes), max(slopes) + 3*np.std(slopes))])#,
                          #method='Powell')

        if result.success:
            optimal_threshold = result.x[0]
            return optimal_threshold
        raise Exception("Optimization failed")
