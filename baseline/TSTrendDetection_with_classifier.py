import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import ks_2samp
from skmultiflow.drift_detection import ADWIN
from xgboost import XGBClassifier
import logging
import torch

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection:
    def __init__(self, bandwidth=1.0, adwin_delta=0.002):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.scaler = StandardScaler()
        self.stats_history = {}  # Хранилище исторических статистик
        self.stats_summary = {}  # Суммарные статистики
        self.max_history = 1000  # Ограничение на размер истории
        self.adwin = {}  # ADWIN для каждой статистики
        self.available_stats = ['alpha', 'variance', 'mse', 'amplitude']
        self.classifier = None  # Для хранения обученной модели классификации

    def fit_mean_shift(self, X, use_clustering=True, stats_to_extract=None):
        """
        Mean Shift model fitting with optional clustering and flexible statistics.
        """
        X = X.copy()

        if isinstance(X, pd.DataFrame):
            X.time = X.time.dt.total_seconds()
            X = X[['time', 'value']].values
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be np.ndarray or pd.DataFrame.")

        if stats_to_extract is None:
            stats_to_extract = self.available_stats
        else:
            stats_to_extract = [s for s in stats_to_extract if s in self.available_stats]
            if not stats_to_extract:
                raise ValueError("No valid statistics specified.")

        for stat in stats_to_extract:
            if stat not in self.stats_history:
                self.stats_history[stat] = []
                self.stats_summary[stat] = {'mean': None, 'std': None}
                self.adwin[stat] = ADWIN(delta=0.002)

        scaler = self.scaler
        X_scaled = scaler.fit_transform(X)

        if use_clustering:
            self.mean_shift.fit(X_scaled)
            labels = self.mean_shift.labels_
        else:
            labels = np.zeros(X.shape[0], dtype=int)

        clusters = np.unique(labels)
        clustered_ts_values = dict()
        clustered_ts_times = dict()
        clustered_stats = dict()

        for cluster in clusters:
            clustered_series = X[np.where(labels == cluster)]
            times, values, stats = self._fit_linear_regression(clustered_series, stats_to_extract)

            clustered_ts_values[cluster] = values
            clustered_ts_times[cluster] = times
            clustered_stats[cluster] = stats

        return (X, labels, clustered_ts_times, clustered_ts_values, clustered_stats)

    def _fit_linear_regression(self, clustered_series, stats_to_extract):
        """
        Fit linear regression and compute selected statistics for a cluster.
        """
        model = LinearRegression()
        X_time = clustered_series[:, 0].reshape(-1, 1)
        y_values = clustered_series[:, 1].reshape(-1, 1)
        model.fit(X_time, y_values)

        clustered_ts_value = model.predict(X_time)
        clustered_ts_time = clustered_series[:, 0]

        stats = {}
        if 'alpha' in stats_to_extract:
            stats['alpha'] = model.coef_[0][0]
        if 'variance' in stats_to_extract:
            stats['variance'] = np.var(clustered_series[:, 1]) if len(clustered_series) > 1 else 0.0
        if 'mse' in stats_to_extract:
            stats['mse'] = mean_squared_error(y_values, clustered_ts_value)
        if 'amplitude' in stats_to_extract:
            stats['amplitude'] = np.max(clustered_series[:, 1]) - np.min(clustered_series[:, 1])

        return clustered_ts_time, clustered_ts_value, stats

    def predict(self, context, model_input, params=None):
        """
        Prediction method using multiple statistics thresholds.
        """
        X, thresholds = model_input
        _, _, _, _, clustered_stats = self.fit_mean_shift(X, stats_to_extract=list(thresholds.keys()))

        anomalies = []
        for cluster_stats in clustered_stats.values():
            anomaly = False
            for stat_name, threshold in thresholds.items():
                if stat_name in cluster_stats and cluster_stats[stat_name] > threshold:
                    anomaly = True
                    break
            anomalies.append(anomaly)
        return np.array(anomalies, dtype=int)

    def downsample(self, X, smoothing_window=None, skip_window=None):
        """
        Method for time series downsampling.
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
        """
        predictions = (values > threshold).astype(int)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        class_weights = {0: 1.0, 1: 5.0 if sum(labels) > 0 else 1.0}
        sample_weights = [class_weights[l] for l in labels]
        bce = log_loss(labels, predictions, sample_weight=sample_weights)
        reg = reg_lambda * (threshold ** 2)
        return bce + reg

    def evaluate_thresholds(self, stats_values, labels, stat_name, use_kfold=True, n_splits=5, reg_lambda=0.01, lr=0.01, max_iter=1000):
        """
        Find the best threshold for a given statistic with optional k-fold cross-validation.

        Parameters
        ----------
        stats_values: np.ndarray
            Array of statistic values.
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        stat_name: str
            Name of the statistic.
        use_kfold: bool
            If True, use k-fold cross-validation; if False, optimize on full data.
        n_splits: int
            Number of folds for cross-validation (if use_kfold=True).
        reg_lambda: float
            Regularization parameter.
        lr: float
            Learning rate for Adam optimizer.
        max_iter: int
            Maximum number of iterations for optimization.

        Returns
        -------
        best_threshold: float
            Optimal threshold for the statistic.
        """
        stats_values = np.array(stats_values)
        labels = np.array(labels)

        if (labels == 1).all():
            return stats_values.min()
        if (labels == 0).all():
            return stats_values.max()     

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

        def optimize_threshold(values, labels, initial_threshold):
            threshold = torch.tensor([initial_threshold], requires_grad=True)
            optimizer = torch.optim.Adam([threshold], lr=lr)

            for _ in range(max_iter):
                optimizer.zero_grad()
                loss = torch.tensor(self._binary_cross_entropy(
                    threshold.item(), values, labels, reg_lambda), requires_grad=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    threshold.clamp_(min=lower_bound, max=upper_bound)

            return threshold.item(), loss.item()

        if use_kfold:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            thresholds = []
            for train_idx, val_idx in kf.split(stats_values):
                train_values, train_labels = stats_values[train_idx], labels[train_idx]
                initial_threshold = np.median(train_values)
                final_threshold, final_loss = optimize_threshold(train_values, train_labels, initial_threshold)
                thresholds.append(final_threshold)
                logging.info(f"Fold threshold for {stat_name}: {final_threshold}, loss: {final_loss}")
            
            if not thresholds:
                logging.warning(f"No successful optimizations for {stat_name}, returning median")
                return np.median(stats_values)
            
            best_threshold = np.mean(thresholds)
        else:
            initial_threshold = np.median(stats_values)
            best_threshold, final_loss = optimize_threshold(stats_values, labels, initial_threshold)
            logging.info(f"Single optimization threshold for {stat_name}: {best_threshold}, loss: {final_loss}")

        logging.info(f"Final threshold for {stat_name}: {best_threshold}")
        return best_threshold

    def train_anomaly_classifier(self, stats_values, labels, classifier_params=None):
        """
        Train an XGBoost classifier to detect anomalies based on statistics.

        Parameters
        ----------
        stats_values: dict
            Dictionary with statistic values (e.g., {'alpha': [...], 'variance': [...]}).
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        classifier_params: dict or None
            Parameters for XGBoost classifier (e.g., {'max_depth': 3}).

        Returns
        -------
        None
            Stores the trained classifier in self.classifier.
        """
        if not stats_values:
            raise ValueError("No statistics provided for training.")
        
        if (labels == 1).all():
            return stats_values.min()
        if (labels == 0).all():
            return stats_values.max()     

        # Prepare features
        stat_names = list(stats_values.keys())
        if not stat_names:
            raise ValueError("No valid statistics provided.")
        
        X = np.array([stats_values[stat_name] for stat_name in stat_names]).T
        y = np.array(labels)

        if len(X) < 5 or len(np.unique(y)) < 2:
            logging.warning("Insufficient data or labels for training classifier")
            self.classifier = None
            return

        # Default parameters with focus on high Recall
        default_params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': 5.0,  # Increase weight for anomalies
            'random_state': 42
        }
        if classifier_params:
            default_params.update(classifier_params)

        # Train XGBoost
        self.classifier = XGBClassifier(**default_params)
        self.classifier.fit(X, y)
        logging.info("Anomaly classifier trained successfully")

    def predict_anomaly_classifier(self, stats_values):
        """
        Predict anomalies using the trained classifier.

        Parameters
        ----------
        stats_values: dict
            Dictionary with statistic values (e.g., {'alpha': [...], 'variance': [...]}).

        Returns
        -------
        np.ndarray
            Predicted labels (1 for anomaly, 0 for normal).
        """
        if self.classifier is None:
            logging.warning("No trained classifier available, returning zeros")
            return np.zeros(len(list(stats_values.values())[0]), dtype=int)

        stat_names = list(stats_values.keys())
        if not stat_names:
            logging.warning("No statistics provided for prediction, returning zeros")
            return np.zeros(len(list(stats_values.values())[0]), dtype=int)

        X = np.array([stats_values[stat_name] for stat_name in stat_names]).T
        predictions = self.classifier.predict(X)
        logging.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
        return predictions

    def detect_stats_drift(self, new_stats, window_size=100, drift_threshold=2.0, ks_pvalue=0.05):
        """
        Detect drift in the distribution of statistics.
        """
        results = {}
        for stat_name in new_stats.keys():
            new_values = np.array(new_stats.get(stat_name, []))
            if len(new_values) == 0:
                results[stat_name] = {'drift_detected': False, 'stats': {}}
                continue

            self.stats_history[stat_name].extend(new_values.tolist())
            if len(self.stats_history[stat_name]) > self.max_history:
                self.stats_history[stat_name] = self.stats_history[-self.max_history:]

            if len(self.stats_history[stat_name]) < window_size or len(new_values) < 10:
                logging.info(f"Insufficient data for {stat_name} drift detection")
                results[stat_name] = {'drift_detected': False, 'stats': {}}
                continue

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
                        logging.info(f"Drift not confirmed for {stat_name}: p-value={p_value:.4f}")
                else:
                    logging.info(f"No significant drift for {stat_name}: z-score={z_score:.2f}")
            else:
                logging.warning(f"Historical std is zero for {stat_name}, skipping z-score check")

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
        """
        results = {}
        for stat_name in new_stats.keys():
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

            for value in new_values:
                self.adwin[stat_name].add_element(value)
                if self.adwin[stat_name].detected_change():
                    drift_detected = True
                    logging.info(f"ADWIN drift detected for {stat_name} at value={value:.4f}, window size={self.adwin[stat_name].width}")

            stats['adwin_window_size'] = self.adwin[stat_name].width
            stats['adwin_mean'] = self.adwin[stat_name].estimation
            stats['drift_detected'] = drift_detected

            self.stats_history[stat_name].extend(new_values.tolist())
            if len(self.stats_history[stat_name]) > self.max_history:
                self.stats_history[stat_name] = self.stats_history[-self.max_history:]

            results[stat_name] = {
                'drift_detected': drift_detected,
                'stats': stats
            }

        return results

    def detect_anomalous_stats(self, stats_values, iqr_multiplier=1.5, z_threshold=3.0):
        """
        Detect anomalous statistics using IQR and Z-score.
        """
        results = {}
        for stat_name in stats_values.keys():
            values = np.array(stats_values.get(stat_name, []))
            if len(values) < 5:
                logging.warning(f"Too few {stat_name} values, returning empty anomaly mask")
                results[stat_name] = {
                    'anomalies': np.zeros_like(values, dtype=bool),
                    'stats': {}
                }
                continue

            q25, q75 = np.percentile(values, [25, 75])
            iqr = q75 - q25
            iqr_lower = q25 - iqr_multiplier * iqr
            iqr_upper = q75 + iqr_multiplier * iqr
            iqr_anomalies = (values < iqr_lower) | (values > iqr_upper)

            mean_value = np.mean(values)
            std_value = np.std(values) if len(values) > 1 else 0.0
            z_scores = np.abs(values - mean_value) / std_value if std_value > 0 else np.zeros_like(values)
            z_anomalies = z_scores > z_threshold

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