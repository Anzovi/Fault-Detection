import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
#from skmultiflow.drift_detection import ADWIN
from xgboost import XGBClassifier
import logging


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection:
    def __init__(self, bandwidth=1.0, adwin_delta=0.002):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.scaler = StandardScaler()
#        self.adwin = {}  # ADWIN для каждой статистики
        self.available_stats = ['alpha', 'variance', 'mse', 'amplitude']
        self.classifier = None  # Для хранения обученной модели классификации

    def fit_mean_shift(self, X, use_clustering=True, stats_to_extract=["alpha"]):
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

        # # Инициализация ADWIN для новых статистик
        # for stat in stats_to_extract:
        #     if stat not in self.adwin:
        #         self.adwin[stat] = ADWIN(delta=0.002)

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

    def evaluate_thresholds(self, stats_matrix, labels, stats_names):
        """
        Find optimal thresholds for each statistic column using Decision Tree.

        Parameters
        ----------
        stats_matrix: np.ndarray
            Matrix where columns are statistics and rows are observations.
        labels: np.ndarray
            Labeled anomalies: 1 if anomaly, 0 otherwise.
        stats_names: list
            Names of statistics corresponding to columns in stats_matrix.

        Returns
        -------
        np.ndarray
            Array of thresholds, one for each column in stats_matrix.
        """
        stats_matrix = np.array(stats_matrix)
        labels = np.array(labels)

        if (labels == 1).all():
            return np.min(stats_matrix, axis=0)
        if (labels == 0).all():
            return np.max(stats_matrix, axis=0)

        # Проверки входных данных
        if len(stats_names) != stats_matrix.shape[1]:
            raise ValueError("stats_names must match the number of columns in stats_matrix.")
        if stats_matrix.shape[0] != len(labels):
            raise ValueError("Number of rows in stats_matrix must match length of labels.")
        if stats_matrix.shape[0] < 2:
            logging.warning("Too few observations, returning median thresholds")
            return np.array([np.median(stats_matrix[:, i]) for i in range(stats_matrix.shape[1])])

        thresholds = []
        for col_idx, stat_name in enumerate(stats_names):
            values = stats_matrix[:, col_idx]

            # Проверка на константные значения или мало данных
            if len(np.unique(values)) < 2:
                logging.warning(f"Constant or insufficient unique values for {stat_name}, using median")
                thresholds.append(np.median(values))
                continue

            # Проверка наличия нормальных и аномальных значений
            normal_values = values[labels == 0]
            anomaly_values = values[labels == 1]
            if len(normal_values) == 0 or len(anomaly_values) == 0:
                logging.warning(f"No normal or anomaly values for {stat_name}, using median")
                thresholds.append(np.median(values))
                continue

            # Обучение решающего дерева
            X = values.reshape(-1, 1)
            y = labels
            clf = DecisionTreeClassifier(max_depth=1, class_weight='balanced', random_state=42)
            clf.fit(X, y)

            # Извлечение порога из корневого узла
            if clf.tree_.node_count > 1:  # Проверка, что дерево имеет разделение
                threshold = clf.tree_.threshold[0]  # Порог корневого узла

                if threshold == -2:  # -2 означает отсутствие разделения
                    logging.warning(f"No valid split for {stat_name}, using median")
                    threshold = np.median(values)
            else:
                logging.warning(f"No split in tree for {stat_name}, using median")
                threshold = np.median(values)

            # Ограничение порога
            threshold = np.clip(threshold, np.min(values), np.max(values))

            # Оценка F1
            predictions = (values > threshold).astype(int)
            f1 = f1_score(labels, predictions)
            logging.info(f"Threshold for {stat_name}: {threshold:.6f}, F1 score: {f1:.4f}")
            thresholds.append(threshold)

        return np.array(thresholds)

    def train_anomaly_classifier(self, stats_values, labels, classifier_params=None):
        """
        Train an XGBoost classifier to detect anomalies based on statistics.
        """

        default_params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': 5.0,
            'random_state': 42
        }
        if classifier_params:
            default_params.update(classifier_params)

        self.classifier = XGBClassifier(**default_params)
        self.classifier.fit(stats_values, labels)
        logging.info("Anomaly classifier trained successfully")


    def predict_anomaly_classifier(self, stats_values):
        """
        Predict anomalies using the trained classifier.
        """
        if self.classifier is None:
            logging.warning("No trained classifier available, returning zeros")

        predictions = self.classifier.predict(stats_values)
        logging.info(f"Predicted {np.sum(predictions)} anomalies out of {len(predictions)} samples")
        return predictions

    def detect_stats_drift(self, stats_matrix, new_stats_matrix, stats_names, window_size=100, drift_threshold=2.0, ks_pvalue=0.05):
        """
        Detect drift in the distribution of statistics between historical and new data.

        Parameters
        ----------
        stats_matrix: np.ndarray
            Historical statistics matrix (rows: observations, columns: stats).
        new_stats_matrix: np.ndarray
            New statistics matrix (rows: observations, columns: stats).
        stats_names: list
            Names of statistics corresponding to columns in stats_matrix.
        window_size: int
            Size of the historical window to consider.
        drift_threshold: float
            Z-score threshold for drift detection.
        ks_pvalue: float
            P-value threshold for KS test.

        Returns
        -------
        dict
            Dictionary with drift detection results for each statistic.
        """
        stats_matrix = np.array(stats_matrix)
        new_stats_matrix = np.array(new_stats_matrix)

        if len(stats_names) != stats_matrix.shape[1] or len(stats_names) != new_stats_matrix.shape[1]:
            raise ValueError("stats_names must match the number of columns in stats_matrix and new_stats_matrix.")
        if stats_matrix.shape[0] < window_size or new_stats_matrix.shape[0] < 10:
            logging.info("Insufficient data for drift detection")
            return {stat_name: {'drift_detected': False, 'stats': {}} for stat_name in stats_names}

        results = {}
        for col_idx, stat_name in enumerate(stats_names):
            historical_values = stats_matrix[-window_size:, col_idx] if stats_matrix.shape[0] > window_size else stats_matrix[:, col_idx]
            new_values = new_stats_matrix[:, col_idx]

            current_mean = np.mean(new_values)
            current_std = np.std(new_values) if len(new_values) > 1 else 0.0
            historical_mean = np.mean(historical_values)
            historical_std = np.std(historical_values) if len(historical_values) > 1 else 0.0

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

            results[stat_name] = {
                'drift_detected': drift_detected,
                'stats': stats
            }

        return results

    def detect_stats_drift_adwin(self, stats_matrix, new_stats_matrix, stats_names):
        """
        Detect drift in statistics using ADWIN.

        Parameters
        ----------
        stats_matrix: np.ndarray
            Historical statistics matrix (rows: observations, columns: stats).
        new_stats_matrix: np.ndarray
            New statistics matrix (rows: observations, columns: stats).
        stats_names: list
            Names of statistics corresponding to columns in stats_matrix.

        Returns
        -------
        dict
            Dictionary with drift detection results for each statistic.
        """
        stats_matrix = np.array(stats_matrix)
        new_stats_matrix = np.array(new_stats_matrix)

        if len(stats_names) != stats_matrix.shape[1] or len(stats_names) != new_stats_matrix.shape[1]:
            raise ValueError("stats_names must match the number of columns in stats_matrix and new_stats_matrix.")
        if new_stats_matrix.shape[0] < 1:
            logging.info("No new data provided for ADWIN drift detection")
            return {stat_name: {'drift_detected': False, 'stats': {}} for stat_name in stats_names}

        results = {}
        for col_idx, stat_name in enumerate(stats_names):
            new_values = new_stats_matrix[:, col_idx]
            drift_detected = False
            stats = {
                'current_mean': np.mean(new_values) if len(new_values) > 0 else 0.0,
                'n_values': len(new_values)
            }

            if stat_name not in self.adwin:
                self.adwin[stat_name] = ADWIN(delta=0.002)

            # Добавляем исторические данные в ADWIN
            historical_values = stats_matrix[:, col_idx]
            for value in historical_values:
                self.adwin[stat_name].add_element(value)

            # Проверяем новые данные
            for value in new_values:
                self.adwin[stat_name].add_element(value)
                if self.adwin[stat_name].detected_change():
                    drift_detected = True
                    logging.info(f"ADWIN drift detected for {stat_name} at value={value:.4f}, window size={self.adwin[stat_name].width}")

            stats['adwin_window_size'] = self.adwin[stat_name].width
            stats['adwin_mean'] = self.adwin[stat_name].estimation
            stats['drift_detected'] = drift_detected

            results[stat_name] = {
                'drift_detected': drift_detected,
                'stats': stats
            }

        return results

    def detect_anomalies(self, stats_matrix, new_stats_matrix, stats_names = None, method='isolation_forest', contamination=0.1, **kwargs):
        """
        Detect anomalies in new_stats_matrix using SOTA anomaly detection methods.

        Parameters
        ----------
        stats_matrix: np.ndarray
            Historical statistics matrix (rows: observations, columns: stats).
        new_stats_matrix: np.ndarray
            New statistics matrix (rows: observations, columns: stats).
        stats_names: list
            Names of statistics corresponding to columns in stats_matrix.
        method: str
            Anomaly detection method: 'isolation_forest', 'one_class_svm', 'lof', 'dbscan'.
        contamination: float
            Expected proportion of anomalies (0.0 to 0.5).
        **kwargs: dict
            Additional parameters for the chosen method.

        Returns
        -------
        dict
            Dictionary with:
            - 'anomalies': np.ndarray of bools (True for anomalies in new_stats_matrix).
            - 'stats': dict with method-specific statistics.
        """
        stats_matrix = np.array(stats_matrix)
        new_stats_matrix = np.array(new_stats_matrix)

        if stats_names:
            if len(stats_names) != stats_matrix.shape[1] or len(stats_names) != new_stats_matrix.shape[1]:
                raise ValueError("stats_names must match the number of columns in stats_matrix and new_stats_matrix.")
        if new_stats_matrix.shape[0] < 5:
            logging.warning("Too few observations in new_stats_matrix, returning no anomalies")
            return {
                'anomalies': np.zeros(new_stats_matrix.shape[0], dtype=bool),
                'stats': {'n_anomalies': 0}
            }

        # Нормализация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(stats_matrix)
        X_test_scaled = scaler.transform(new_stats_matrix)

        # Выбор метода детекции
        if method == 'isolation_forest':
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                **kwargs
            )
            model.fit(X_train_scaled)
            predictions = model.predict(X_test_scaled)
            anomalies = predictions == -1

        elif method == 'one_class_svm':
            model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                **kwargs
            )
            model.fit(X_train_scaled)
            predictions = model.predict(X_test_scaled)
            anomalies = predictions == -1

        elif method == 'lof':
            model = LocalOutlierFactor(
                n_neighbors=min(20, new_stats_matrix.shape[0] - 1),
                contamination=contamination,
                novelty=True,
                **kwargs
            )
            model.fit(X_train_scaled)
            predictions = model.predict(X_test_scaled)
            anomalies = predictions == -1

        elif method == 'dbscan':
            model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5),
                **kwargs
            )
            predictions = model.fit_predict(X_test_scaled)
            anomalies = predictions == -1

        else:
            raise ValueError(f"Unsupported method: {method}")

        # Статистики
        n_anomalies = np.sum(anomalies)
        stats = {
            'n_anomalies': n_anomalies,
            'method': method,
            'contamination': contamination
        }

        # Логирование
        logging.info(f"Anomalies detected: {n_anomalies} out of {new_stats_matrix.shape[0]} samples using {method}")

        return {
            'anomalies': anomalies,
            'stats': stats
        }
