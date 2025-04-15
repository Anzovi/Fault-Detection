import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
from skmultiflow.drift_detection import ADWIN
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection:
    def __init__(self, bandwidth=1.0, adwin_delta=0.002):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.scaler = StandardScaler()
        self.stats_history = {
            'alpha': [],
            'variance': [],
            'mse': [],
            'amplitude': []
        }  # Хранилище исторических статистик
        self.stats_summary = {
            'alpha': {'mean': None, 'std': None},
            'variance': {'mean': None, 'std': None},
            'mse': {'mean': None, 'std': None},
            'amplitude': {'mean': None, 'std': None}
        }  # Суммарные статистики
        self.max_history = 1000  # Ограничение на размер истории
        self.adwin = {
            'alpha': ADWIN(delta=adwin_delta),
            'variance': ADWIN(delta=adwin_delta),
            'mse': ADWIN(delta=adwin_delta),
            'amplitude': ADWIN(delta=adwin_delta)
        }  # ADWIN для каждой статистики

    def fit_mean_shift(self, X, use_clustering=True):
        """
        Mean Shift model fitting with optional clustering and extended statistics.

        Parameters
        ----------
        X: np.ndarray (shape = (samples_count, 2)) :
            first samples timedelta from first sample
            second samples values
        or
        X: pd.DataFrame:
            has column timedelta from first sample
            has column value
        use_clustering: bool
            If True, use MeanShift clustering; if False, treat all data as one cluster.

        Returns
        -------
        X: np.ndarray (shape = (samples_count, 2)) :
            first samples time
            second samples values
        labels: np.ndarray:
            np.ndarray (shape = (samples_count,))
        clustered_ts_times: dict:
            dict of np.ndarray of clustered times
        clustered_ts_values: dict:
            dict of np.ndarray of clustered values
        clustered_stats: dict:
            dict of statistics for each cluster: {'alpha', 'variance', 'mse', 'amplitude'}
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

        # Clustering or single cluster
        if use_clustering:
            self.mean_shift.fit(X_scaled)
            labels = self.mean_shift.labels_
        else:
            labels = np.zeros(X.shape[0], dtype=int)  # Все точки в одном кластере

        # Process clusters
        clusters = np.unique(labels)
        clustered_ts_values = dict()
        clustered_ts_times = dict()
        clustered_stats = dict()

        for cluster in clusters:
            clustered_series = X[np.where(labels == cluster)]
            times, values, stats = self._fit_linear_regression(clustered_series)

            clustered_ts_values[cluster] = values
            clustered_ts_times[cluster] = times
            clustered_stats[cluster] = stats

        return (X, labels, clustered_ts_times, clustered_ts_values, clustered_stats)

    def _fit_linear_regression(self, clustered_series):
        """
        Fit linear regression and compute extended statistics for a cluster.

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
        stats: dict:
            Statistics: {'alpha', 'variance', 'mse', 'amplitude'}
        """
        model = LinearRegression()
        X_time = clustered_series[:, 0].reshape(-1, 1)
        y_values = clustered_series[:, 1].reshape(-1, 1)
        model.fit(X_time, y_values)

        clustered_ts_value = model.predict(X_time)
        clustered_ts_time = clustered_series[:, 0]

        # Compute statistics
        alpha = model.coef_[0]
        variance = np.var(clustered_series[:, 1]) if len(clustered_series) > 1 else 0.0
        mse = mean_squared_error(y_values, clustered_ts_value)
        amplitude = np.max(clustered_series[:, 1]) - np.min(clustered_series[:, 1])

        stats = {
            'alpha': alpha,
            'variance': variance,
            'mse': mse,
            'amplitude': amplitude
        }

        return clustered_ts_time, clustered_ts_value, stats

    def predict(self, context, model_input, params=None):
        """
        Prediction method using multiple statistics thresholds.

        Parameters:
        -----------
        context : Any
            Ignored.
        model_input : tuple
            (DataFrame or ndarray, dict of thresholds {'alpha', 'variance', 'mse', 'amplitude'}).
        params : dict, optional
            Ignored.

        Returns:
        --------
        np.ndarray
            Boolean array indicating anomalies based on any statistic exceeding its threshold.
        """
        X, thresholds = model_input
        _, _, _, _, clustered_stats = self.fit_mean_shift(X)

        anomalies = []
        for cluster_stats in clustered_stats.values():
            anomaly = False
            for stat_name, threshold in thresholds.items():
                if cluster_stats[stat_name] > threshold:
                    anomaly = True
                    break
            anomalies.append(anomaly)
        return np.array(anomalies, dtype=int)

    def downsample(self, X, smoothing_window=None, skip_window=None):
        """
        Parameters and Returns unchanged.
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

    def _binary_cross_entropy(self, threshold, values, labels, reg_lambda=0.01):
        """
        Binary cross entropy with class weights.

        Parameters
        ----------
        threshold: float
            Threshold for classifying values.
        values: np.ndarray
            Array of statistic values.
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        reg_lambda: float
            Regularization parameter.

        Returns
        -------
        float
            Weighted binary cross entropy with regularization.
        """
        predictions = (values > threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        class_weights = {0: 1.0, 1: 5.0 if sum(labels) > 0 else 1.0}  # Prioritize anomalies
        sample_weights = [class_weights[l] for l in labels]
        bce = log_loss(labels, predictions, sample_weight=sample_weights)
        reg = reg_lambda * (threshold ** 2)
        return bce + reg

    def evaluate_thresholds(self, stats_values, labels, stat_name, n_splits=5, reg_lambda=0.01):
        """
        Find the best threshold for a given statistic using k-fold cross-validation.

        Parameters
        ----------
        stats_values: np.ndarray
            Array of statistic values (e.g., alpha, variance, mse, amplitude).
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        stat_name: str
            Name of the statistic ('alpha', 'variance', 'mse', 'amplitude').
        n_splits: int
            Number of folds for cross-validation.
        reg_lambda: float
            Regularization parameter.

        Returns
        -------
        best_threshold: float
            Optimal threshold for the statistic.
        """
        stats_values = np.array(stats_values)
        labels = np.array(labels)

        if len(stats_values) < 5:
            logging.warning(f"Too few {stat_name} values, returning median as threshold")
            return np.median(stats_values)

        # Compute bounds using IQR
        q25, q75 = np.percentile(stats_values, [25, 75])
        iqr = q75 - q25
        lower_bound = max(min(stats_values), q25 - 1.5 * iqr)
        upper_bound = min(max(stats_values), q75 + 1.5 * iqr)
        if upper_bound - lower_bound < 1e-5:
            upper_bound = lower_bound + 0.1 * (max(stats_values) - min(stats_values))

        # K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        thresholds = []
        for train_idx, val_idx in kf.split(stats_values):
            train_values, train_labels = stats_values[train_idx], labels[train_idx]
            initial_threshold = np.median(train_values)
            result = minimize(self._binary_cross_entropy, initial_threshold,
                             args=(train_values, train_labels, reg_lambda),
                             method='L-BFGS-B',
                             bounds=[(lower_bound, upper_bound)])
            if result.success:
                thresholds.append(result.x[0])
                logging.info(f"Fold threshold for {stat_name}: {result.x[0]}, loss: {result.fun}")
            else:
                logging.warning(f"Optimization failed for {stat_name} fold, skipping")

        if not thresholds:
            logging.warning(f"No successful optimizations for {stat_name}, returning median")
            return np.median(stats_values)

        best_threshold = np.mean(thresholds)
        logging.info(f"Final threshold for {stat_name}: {best_threshold}")
        return best_threshold

    def detect_stats_drift(self, new_stats, window_size=100, drift_threshold=2.0, ks_pvalue=0.05):
        """
        Detect drift in the distribution of statistics (alpha, variance, mse, amplitude).

        Parameters
        ----------
        new_stats: dict
            Dictionary with new statistic values {'alpha', 'variance', 'mse', 'amplitude'}.
        window_size: int
            Size of the historical window.
        drift_threshold: float
            Number of standard deviations to consider as drift.
        ks_pvalue: float
            P-value threshold for KS-test.

        Returns
        -------
        dict
            Dictionary with drift detection results for each statistic.
        """
        results = {}
        for stat_name in ['alpha', 'variance', 'mse', 'amplitude']:
            new_values = np.array(new_stats.get(stat_name, []))
            if len(new_values) == 0:
                results[stat_name] = {'drift_detected': False, 'stats': {}}
                continue

            # Update history
            self.stats_history[stat_name].extend(new_values.tolist())
            if len(self.stats_history[stat_name]) > self.max_history:
                self.stats_history[stat_name] = self.stats_history[stat_name][-self.max_history:]

            # Check if enough data
            if len(self.stats_history[stat_name]) < window_size or len(new_values) < 10:
                logging.info(f"Insufficient data for {stat_name} drift detection")
                results[stat_name] = {'drift_detected': False, 'stats': {}}
                continue

            # Compare with historical data
            historical_values = np.array(self.stats_history[stat_name][-window_size:])
            current_mean = np.mean(new_values)
            current_std = np.std(new_values) if len(new_values) > 1 else 0.0

            if self.stats_summary[stat_name]['mean'] is None:
                self.stats_summary[stat_name]['mean'] = np.mean(historical_values)
                self.stats_summary[stat_name]['std'] = np.std(historical_values) if len(historical_values) > 1 else 0.0
            historical_mean = self.stats_summary[stat_name]['mean']
            historical_std = self.stats_summary[stat_name]['std']

            drift_detected = False
            stats = {
                'current_mean': current_mean,
                'current_std': current_std,
                'historical_mean': historical_mean,
                'historical_std': historical_std
            }

            if historical_std > 0:
                z_score = abs(current_mean - historical_mean) / historical_std
                if z_score > drift_threshold:
                    ks_stat, p_value = ks_2samp(historical_values, new_values)
                    if p_value < ks_pvalue:
                        drift_detected = True
                        logging.info(f"Drift detected for {stat_name}: z-score={z_score:.2f}, KS p-value={p_value:.4f}")
                    else:
                        logging.info(f"Drift not confirmed for {stat_name} by KS-test: p-value={p_value:.4f}")
                else:
                    logging.info(f"No significant drift for {stat_name}: z-score={z_score:.2f}")
            else:
                logging.warning(f"Historical std is zero for {stat_name}, skipping z-score check")

            # Update summary statistics
            self.stats_summary[stat_name]['mean'] = 0.9 * self.stats_summary[stat_name]['mean'] + 0.1 * current_mean
            self.stats_summary[stat_name]['std'] = 0.9 * self.stats_summary[stat_name]['std'] + 0.1 * current_std

            results[stat_name] = {
                'drift_detected': drift_detected,
                'stats': stats
            }

        return results

    def detect_stats_drift_adwin(self, new_stats):
        """
        Detect drift in statistics using ADWIN.

        Parameters
        ----------
        new_stats: dict
            Dictionary with new statistic values {'alpha', 'variance', 'mse', 'amplitude'}.

        Returns
        -------
        dict
            Dictionary with drift detection results for each statistic.
        """
        results = {}
        for stat_name in ['alpha', 'variance', 'mse', 'amplitude']:
            new_values = np.array(new_stats.get(stat_name, []))
            drift_detected = False
            stats = {
                'current_mean': np.mean(new_values) if len(new_values) > 0 else 0.0,
                'n_values': len(new_values)
            }

            if len(new_values) < 1:
                logging.info(f"No new {stat_name} provided for ADWIN drift detection")
                results[stat_name] = {'drift_detected': False, 'stats': stats}
                continue

            # Update ADWIN
            for value in new_values:
                self.adwin[stat_name].add_element(value)
                if self.adwin[stat_name].detected_change():
                    drift_detected = True
                    logging.info(f"ADWIN drift detected for {stat_name} at value={value:.4f}, window size={self.adwin[stat_name].width}")

            # Update stats
            stats['adwin_window_size'] = self.adwin[stat_name].width
            stats['adwin_mean'] = self.adwin[stat_name].estimation
            stats['drift_detected'] = drift_detected

            # Update history
            self.stats_history[stat_name].extend(new_values.tolist())
            if len(self.stats_history[stat_name]) > self.max_history:
                self.stats_history[stat_name] = self.stats_history[stat_name][-self.max_history:]

            results[stat_name] = {
                'drift_detected': drift_detected,
                'stats': stats
            }

        return results

    def detect_anomalous_stats(self, stats_values, iqr_multiplier=1.5, z_threshold=3.0):
        """
        Detect anomalous statistics using IQR and Z-score.

        Parameters
        ----------
        stats_values: dict
            Dictionary with statistic values {'alpha', 'variance', 'mse', 'amplitude'}.
        iqr_multiplier: float
            Multiplier for IQR.
        z_threshold: float
            Threshold for Z-score.

        Returns
        -------
        dict
            Dictionary with anomaly masks and stats for each statistic.
        """
        results = {}
        for stat_name in ['alpha', 'variance', 'mse', 'amplitude']:
            values = np.array(stats_values.get(stat_name, []))
            if len(values) < 5:
                logging.warning(f"Too few {stat_name} values, returning empty anomaly mask")
                results[stat_name] = {
                    'anomalies': np.zeros_like(values, dtype=bool),
                    'stats': {}
                }
                continue

            # IQR method
            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            iqr_lower = q25 - iqr_multiplier * iqr
            iqr_upper = q75 + iqr_multiplier * iqr
            iqr_anomalies = (values < iqr_lower) | (values > iqr_upper)

            # Z-score method
            mean_value = np.mean(values)
            std_value = np.std(values) if len(values) > 1 else 0.0
            z_scores = np.abs(values - mean_value) / std_value if std_value > 0 else np.zeros_like(values)
            z_anomalies = z_scores > z_threshold

            # Combine
            anomalies = iqr_anomalies | z_anomalies

            stats = {
                'mean_value': mean_value,
                'std_value': std_value,
                'iqr_lower': iqr_lower,
                'iqr_upper': iqr_upper,
                'n_anomalies': np.sum(anomalies)
            }
            logging.info(f"Anomalies detected for {stat_name}: {stats['n_anomalies']} out of {len(values)}")
            results[stat_name] = {
                'anomalies': anomalies,
                'stats': stats
            }

        return results