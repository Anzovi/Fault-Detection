import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.scaler = StandardScaler()
        self.alpha_history = []  # Хранилище исторических alpha
        self.alpha_stats = {'mean': None, 'std': None}  # Статистики alpha
        self.max_history = 1000  # Ограничение на размер истории

    def fit_mean_shift(self, X):
        """
        Mean Shift model fitting.
        Parameters and Returns unchanged (see original).
        """
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X.time = X.time.dt.total_seconds()
            X = X[['time', 'value']].values
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be np.ndarray or pd.DataFrame.")

        scaler = self.scaler
        X_scaled = scaler.fit_transform(X)

        self.mean_shift.fit(X_scaled)
        labels = self.mean_shift.labels_

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
        Parameters and Returns unchanged (see original).
        """
        model = LinearRegression()
        model.fit(clustered_series[:, 0].reshape(-1, 1),
                  clustered_series[:, 1].reshape(-1, 1))

        clustered_ts_value = model.predict(clustered_series[:, 0].reshape(-1, 1))
        clustered_ts_time = clustered_series[:, 0]
        clustered_ts_cofs = model.coef_[0]
        clustered_ts_intercepts = model.intercept_

        return (clustered_ts_time, clustered_ts_value,
                clustered_ts_cofs, clustered_ts_intercepts)

    def predict(self, context, model_input, params=None):
        """
        Parameters and Returns unchanged (see original).
        """
        return self._predict_internal(model_input[0], model_input[1])

    def _predict_internal(self, X, alpha):
        """
        Parameters and Returns unchanged (see original).
        """
        _, _, _, _, clustered_ts_cofs, _ = self.fit_mean_shift(X)
        return (np.array(list(clustered_ts_cofs.values())) > alpha).astype(int)

    def downsample(self, X, smoothing_window=None, skip_window=None):
        """
        Parameters and Returns unchanged (see original).
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
        Parameters and Returns unchanged (see original).
        """
        predictions = (slopes > threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        class_weights = {0: 1.0, 1: 5.0 if sum(labels) > 0 else 1.0}
        sample_weights = [class_weights[l] for l in labels]
        bce = log_loss(labels, predictions, sample_weight=sample_weights)
        reg = reg_lambda * (threshold ** 2)
        return bce + reg

    def evaluate_thresholds(self, slopes, labels, n_splits=5, reg_lambda=0.01):
        """
        Parameters and Returns unchanged (see original).
        """
        slopes = np.array(slopes)
        labels = np.array(labels)

        if len(slopes) < 5:
            logging.warning("Too few slopes, returning median as threshold")
            return np.median(slopes)

        q25, q75 = np.percentile(slopes, [25, 75])
        iqr = q75 - q25
        lower_bound = max(min(slopes), q25 - 1.5 * iqr)
        upper_bound = min(max(slopes), q75 + 1.5 * iqr)
        if upper_bound - lower_bound < 1e-5:
            upper_bound = lower_bound + 0.1 * (max(slopes) - min(slopes))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        thresholds = []
        for train_idx, val_idx in kf.split(slopes):
            train_slopes, train_labels = slopes[train_idx], labels[train_idx]
            initial_threshold = np.median(train_slopes)
            result = minimize(self._binary_cross_entropy, initial_threshold,
                             args=(train_slopes, train_labels, reg_lambda),
                             method='L-BFGS-B',
                             bounds=[(lower_bound, upper_bound)])
            if result.success:
                thresholds.append(result.x[0])
                logging.info(f"Fold threshold: {result.x[0]}, loss: {result.fun}")
            else:
                logging.warning("Optimization failed for fold, skipping")

        if not thresholds:
            logging.warning("No successful optimizations, returning median as threshold")
            return np.median(slopes)

        best_threshold = np.mean(thresholds)
        logging.info(f"Final threshold: {best_threshold}")
        return best_threshold

    def detect_alpha_drift(self, new_alphas, window_size=100, drift_threshold=2.0, ks_pvalue=0.05):
        """
        Detect drift in the distribution of alpha values.

        Parameters
        ----------
        new_alphas: np.ndarray
            Array of new alpha values (slopes) to check for drift.
        window_size: int
            Size of the historical window to compare against.
        drift_threshold: float
            Number of standard deviations to consider as drift.
        ks_pvalue: float
            P-value threshold for KS-test to confirm drift.

        Returns
        -------
        bool
            True if drift detected, False otherwise.
        dict
            Statistics of the current and historical alpha distributions.
        """
        new_alphas = np.array(new_alphas)

        # Обновление истории
        self.alpha_history.extend(new_alphas.tolist())
        if len(self.alpha_history) > self.max_history:
            self.alpha_history = self.alpha_history[-self.max_history:]

        # Если недостаточно данных, дрейф не проверяется
        if len(self.alpha_history) < window_size or len(new_alphas) < 10:
            logging.info("Insufficient data for drift detection")
            return False, {}

        # Разделение на историческое и новое окно
        historical_alphas = np.array(self.alpha_history[-window_size:])
        current_mean = np.mean(new_alphas)
        current_std = np.std(new_alphas) if len(new_alphas) > 1 else 0.0

        # Обновление статистик
        if self.alpha_stats['mean'] is None:
            self.alpha_stats['mean'] = np.mean(historical_alphas)
            self.alpha_stats['std'] = np.std(historical_alphas) if len(historical_alphas) > 1 else 0.0
        historical_mean = self.alpha_stats['mean']
        historical_std = self.alpha_stats['std']

        # Проверка дрейфа по среднему
        drift_detected = False
        if historical_std > 0:
            z_score = abs(current_mean - historical_mean) / historical_std
            if z_score > drift_threshold:
                # Подтверждение дрейфа с помощью KS-теста
                ks_stat, p_value = ks_2samp(historical_alphas, new_alphas)
                if p_value < ks_pvalue:
                    drift_detected = True
                    logging.info(f"Drift detected: z-score={z_score:.2f}, KS p-value={p_value:.4f}")
                else:
                    logging.info(f"Drift not confirmed by KS-test: p-value={p_value:.4f}")
            else:
                logging.info(f"No significant drift: z-score={z_score:.2f}")
        else:
            logging.warning("Historical std is zero, skipping z-score check")

        # Обновление статистик
        self.alpha_stats['mean'] = 0.9 * self.alpha_stats['mean'] + 0.1 * current_mean
        self.alpha_stats['std'] = 0.9 * self.alpha_stats['std'] + 0.1 * current_std

        stats = {
            'current_mean': current_mean,
            'current_std': current_std,
            'historical_mean': historical_mean,
            'historical_std': historical_std,
            'drift_detected': drift_detected
        }
        return drift_detected, stats

    def detect_anomalous_alpha(self, alphas, iqr_multiplier=1.5, z_threshold=3.0):
        """
        Detect anomalous alpha values using IQR and Z-score.

        Parameters
        ----------
        alphas: np.ndarray
            Array of alpha values to check for anomalies.
        iqr_multiplier: float
            Multiplier for IQR to define anomaly bounds.
        z_threshold: float
            Threshold for Z-score to define anomalies.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates an anomalous alpha.
        dict
            Statistics used for anomaly detection.
        """
        alphas = np.array(alphas)
        if len(alphas) < 5:
            logging.warning("Too few alphas, returning empty anomaly mask")
            return np.zeros_like(alphas, dtype=bool), {}

        # IQR-метод
        q25, q75 = np.percentile(alphas, [25, 75])
        iqr = q75 - q25
        iqr_lower = q25 - iqr_multiplier * iqr
        iqr_upper = q75 + iqr_multiplier * iqr
        iqr_anomalies = (alphas < iqr_lower) | (alphas > iqr_upper)

        # Z-score метод
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas) if len(alphas) > 1 else 0.0
        z_scores = np.abs(alphas - mean_alpha) / std_alpha if std_alpha > 0 else np.zeros_like(alphas)
        z_anomalies = z_scores > z_threshold

        # Комбинируем: аномалия, если подтверждается хотя бы одним методом (для высокого Recall)
        anomalies = iqr_anomalies | z_anomalies

        stats = {
            'mean_alpha': mean_alpha,
            'std_alpha': std_alpha,
            'iqr_lower': iqr_lower,
            'iqr_upper': iqr_upper,
            'n_anomalies': np.sum(anomalies)
        }
        logging.info(f"Anomalies detected: {stats['n_anomalies']} out of {len(alphas)}")
        return anomalies, stats