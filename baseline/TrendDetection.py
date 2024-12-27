import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


class TSTrendDetection:
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
            first samples time
            second samples values
        or
        X: pd.DataFrame:
            has column time
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
        
        if isinstance(X, pd.DataFrame):
            X = X[['time', 'value']].copy().values
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be np.ndarray or pd.DataFrame.")

        # Scaling ts data
        scaler = self.scaler
        X_scaled = scaler.fit_transform(X)

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

    def predict(self, X):
        """
        Prediction using slopes of linear regressions.
        if true then anomaly detected in time series
        """

        _, _, _, _, clustered_ts_cofs, _ = self.fit_mean_shift(X)

        return (clustered_ts_cofs > alpha).any()

    def downsample(self, time_series, smoothing_window, skip_window):
        """
        Method for time series downsampling.
        """
        if not isinstance(time_series, pd.Series):
            raise ValueError("Time series mus be a pd.Series.")

        time_series = time_series.copy().rolling(window=smoothing_window,
                                                 min_periods=1).median()

        time_series.index = pd.to_timedelta(time_series.index, unit='s')
        time_series = time_series.resample(skip_window).median()


        return pd.DataFrame({'time': time_series.index,
                             'value': time_series.values})

    def _binary_cross_entropy(self, threshold, slopes, labels):
        """
        binary cross entropy
        ----------
        threshold: float:
            
        slopes: list, array:
            list of slopes, used as thresholds
        labels: np.ndarray (shape = (samples of cluster, 2)) :
            labeled anomalies: 1 if anomaly, 0 otherwise
        Returns
        -------
        best_threshold: float:
            threshold corresponding to highest f1 score
        """
        predictions = (slopes >= threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        bce = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        return bce

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
          raise ValueError("Thresholds list must consist of at least two elements.")

        initial_threshold = np.mean(slopes)

        # binary cross entropy minimization
        result = minimize(self._binary_cross_entropy, initial_threshold, args=(slopes, labels), bounds=[(min(slopes), max(slopes))])
        
        if result.success:
            optimal_threshold = result.x[0]
            return optimal_threshold
        raise Exception("Optimization failed")

    def plot_clustering_results(self, X, labels):
        """A method for graphical display of clustering results."""
        plt.plot(X[:, 0], X[:, 1], label='Time Series Data', color='blue')
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title('Clustering results using Mean Shift')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.colorbar(label='Метки кластеров')
        plt.grid()
        plt.show()

    def plot_LR_results(self, X, labels, clustered_times, clustered_values):
        """
        A method for graphical display of 
        the results of constructing a linear regression.
        """
        plt.plot(X[:, 0], X[:, 1], label='Time Series Data', color='blue')
        for cluster in np.unique(labels):
          plt.plot(clustered_times[cluster], clustered_values[cluster],
                   label='Linear Trend', color='red')
        plt.legend()
        plt.grid()
        plt.show()
