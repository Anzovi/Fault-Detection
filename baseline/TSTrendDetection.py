import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection():
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
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
        clustered_ts_value: dict:
            dict of np.ndarray of clustered values
        clustered_ts_time: dict:
            dict of np.ndarray of clustered times
        clustered_ts_cofs: dict:
            dict of slopes of linear regression
        clustered_ts_intercepts: dict:
            dict of intercepts of linear regression
        """
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X.time = X.time.dt.total_seconds()
            X = X[['time', 'value']].values
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

        clustered_ts_values = dict()
        clustered_ts_times = dict()
        clustered_ts_cofs = dict()
        clustered_ts_intercepts = dict()

        for cluster in clusters:
            clustered_series = X[np.where(labels == cluster)]
            times, values, cof, intercept = self._fit_linear_regression(clustered_series)

            clustered_ts_values[cluster] = values
            clustered_ts_times[cluster] = times
            clustered_ts_cofs[cluster] = cof[0]
            clustered_ts_intercepts[cluster] = intercept

        return (X, labels, clustered_ts_times, clustered_ts_values,
                clustered_ts_cofs, clustered_ts_intercepts)

    def _fit_linear_regression(self, clustered_series):
        """
        Parameters
        ----------
        clustered_series: np.ndarray (shape = (samples of cluster, 2)) :
            first samples time
            second samples values

        Returns
        -------
        clustered_ts_time: np.ndarray:
            np.ndarray of clustered times
        clustered_ts_value: np.ndarray:
            np.ndarray of clustered values
        clustered_ts_cofs: float:
            slope of a linear regression
        clustered_ts_intercepts: float:
            intercept of a linear regression
        """
        model = LinearRegression()
        model.fit(clustered_series[:, 0].reshape(-1, 1),
                  clustered_series[:, 1].reshape(-1, 1))

        clustered_ts_value = model.predict(clustered_series[:, 0].reshape(-1, 1))
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
            The input DataFrame or ndarray classified as normal or has anomaly slope, and alpha threshold.
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
        return (np.array(list(clustered_ts_cofs.values())) > alpha).astype(int)

    def downsample(self, X, smoothing_window=None, skip_window=None):
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
            raise ValueError("Time series must be a pd.DataFrame.")
        X = X.copy()
        if smoothing_window is not None:
            X.value = X.value.rolling(window=smoothing_window, min_periods=1).median()

        if skip_window is not None:
            X.index = X.time
            X = X.value.resample(skip_window).median().fillna(0)
            X = pd.DataFrame({'time': X.index, 'value': X.values})
        return X

    def _binary_cross_entropy(self, threshold, slopes, labels, reg_lambda=0.01):
        """
        Binary cross entropy with class weights to prioritize anomaly detection.

        Parameters
        ----------
        threshold: float
            Threshold for classifying slopes.
        slopes: np.ndarray
            Array of slopes used as thresholds.
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        reg_lambda: float
            Regularization parameter to penalize extreme thresholds.

        Returns
        -------
        float
            Weighted binary cross entropy with regularization.
        """
        predictions = (slopes > threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        # Увеличиваем вес аномалий для минимизации пропусков
        class_weights = {0: 1.0, 1: 5.0 if sum(labels) > 0 else 1.0}  # Вес аномалий выше
        sample_weights = [class_weights[l] for l in labels]
        bce = log_loss(labels, predictions, sample_weight=sample_weights)
        reg = reg_lambda * (threshold ** 2)  # Регуляризация
        return bce + reg

    def evaluate_thresholds(self, slopes, labels, n_splits=5, reg_lambda=0.01):
        """
        Find the best threshold using k-fold cross-validation to minimize binary cross entropy,
        prioritizing high recall to reduce missed anomalies.

        Parameters
        ----------
        slopes: np.ndarray
            Array of slopes of linear regressions.
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        n_splits: int
            Number of folds for cross-validation.
        reg_lambda: float
            Regularization parameter.

        Returns
        -------
        best_threshold: float
            Threshold corresponding to best cross entropy result.
        """
        slopes = np.array(slopes)
        labels = np.array(labels)

        # Проверка на малое количество данных
        if len(slopes) < 5:
            logging.warning("Too few slopes, returning median as threshold")
            return np.median(slopes)

        if (labels == 1).all():
            return slopes.min()
        if (labels == 0).all():
            return slopes.max()

        # Вычисление границ на основе IQR
        q25, q75 = np.percentile(slopes, [25, 75])
        iqr = q75 - q25
        lower_bound = max(min(slopes), q25 - 1.5 * iqr)
        upper_bound = min(max(slopes), q75 + 1.5 * iqr)
        if upper_bound - lower_bound < 1e-5:  # Обеспечение минимальной ширины
            upper_bound = lower_bound + 0.1 * (max(slopes) - min(slopes))

        # Кросс-валидация
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        thresholds = []
        for train_idx, val_idx in kf.split(slopes):
            train_slopes, train_labels = slopes[train_idx], labels[train_idx]
            initial_threshold = np.median(train_slopes)  # Медиана для устойчивости
            result = minimize(self._binary_cross_entropy, initial_threshold,
                             args=(train_slopes, train_labels, reg_lambda),
                             method='L-BFGS-B',
                             bounds=[(lower_bound, upper_bound)])
            if result.success:
                thresholds.append(result.x[0])
                logging.info(f"Fold threshold: {result.x[0]}, loss: {result.fun}")
            else:
                logging.warning("Optimization failed for fold, skipping")

        # Если оптимизация не удалась ни в одном фолде, возвращаем медиану
        if not thresholds:
            logging.warning("No successful optimizations, returning median as threshold")
            return np.median(slopes)

        best_threshold = np.mean(thresholds)
        logging.info(f"Final threshold: {best_threshold}")
        return best_threshold