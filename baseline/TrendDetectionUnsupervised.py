import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, KMeans
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, f1_score, silhouette_score
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
import logging
import pywt
import pymannkendall as mk
from scipy import stats
from scipy.signal import welch


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTrendDetection:
    def __init__(self, bandwidth=1.0, use_kmeans=True):
        self.bandwidth = bandwidth
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.scaler = StandardScaler()
        self.available_stats = [
            'alpha', 'variance', 'mse', 'amplitude',
            'skewness', 'kurtosis', 'abs_energy',
            'fft_amp1', 'fft_amp2', 'hurst', 'ac1', 'stl_slope',
            'slope_linreg', 'p_linreg', 'r2_linreg', 'mean',
            'longest_above', 'longest_below',
            'mk_trend', 'mk_p', 'ts_slope',
            ]

        self.classifier = None  # Для хранения обученной модели классификации
        self.use_kmeans = use_kmeans

    def fit_mean_shift(self, X, use_clustering=True, stats_to_extract=None, max_k=15):
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

        scaler = self.scaler
        X_scaled = scaler.fit_transform(X)

        if use_clustering:
            if self.use_kmeans:
                best_score = -1
                best_k = 2
                best_labels = None
                for k in range(2, max_k + 1):
                    kmeans = KMeans(n_clusters=k, random_state=11)
                    labels = kmeans.fit_predict(X_scaled)
                    try:
                        score = silhouette_score(X_scaled, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                            best_labels = labels
                    except:
                        continue

                labels = best_labels
            else:
                mean_shift = MeanShift(bandwidth=self.bandwidth)
                labels = mean_shift.fit_predict(X_scaled)
        else:
            labels = np.zeros(X.shape[0], dtype=int)

        clusters = np.unique(labels)
        clustered_ts_values = dict()
        clustered_ts_times = dict()
        clustered_stats = dict()

        for cluster in clusters:
            clustered_series = X[labels == cluster]
            times, values, stats = self.extract_segment_features(clustered_series, stats_to_extract)

            clustered_ts_values[cluster] = values
            clustered_ts_times[cluster] = times
            clustered_stats[cluster] = stats

        return (X, labels, clustered_ts_times, clustered_ts_values, clustered_stats)

    def extract_segment_features(self, clustered_series: np.ndarray, stats_to_extract=None):
        """
        blank
        """
        if stats_to_extract is None:
            stats_to_extract = self.available_stats

        times = clustered_series[:, 0]
        values = clustered_series[:, 1]
        stats_dict = {}

        # Линейная регрессия
        model = LinearRegression()
        X_time = times.reshape(-1, 1)
        y_values = values.reshape(-1, 1)
        model.fit(X_time, y_values)
        preds = model.predict(X_time).flatten()


        # slope, p-value, r2 линейной регрессии
        if {'alpha', 'p_linreg', 'r2_linreg'} & set(stats_to_extract):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, values)
                if 'alpha' in stats_to_extract:
                    stats_dict['alpha'] = slope
                if 'p_linreg' in stats_to_extract:
                    stats_dict['p_linreg'] = p_value
                if 'r2_linreg' in stats_to_extract:
                    stats_dict['r2_linreg'] = r_value**2
            except Exception:
                stats_dict.update({k: np.nan for k in ['slope_linreg', 'p_linreg', 'r2_linreg'] if k in stats_to_extract})

        # mean, variance, mse, skewness, kurtosis
        if 'mean' in stats_to_extract:
            stats_dict['mean'] = np.mean(values)
        if 'variance' in stats_to_extract:
            stats_dict['variance'] = np.var(values)
        if 'mse' in stats_to_extract:
            stats_dict['mse'] = mean_squared_error(y_values, preds)
        if 'skewness' in stats_to_extract:
            stats_dict['skewness'] = stats.skew(values)
        if 'kurtosis' in stats_to_extract:
            stats_dict['kurtosis'] = stats.kurtosis(values)

        # longest strike above / below mean
        if 'longest_above' in stats_to_extract or 'longest_below' in stats_to_extract:
            try:
                import itertools
                mean_val = np.mean(values)
                above = values > mean_val
                if 'longest_above' in stats_to_extract:
                    stats_dict['longest_above'] = max((sum(1 for _ in g) for k, g in itertools.groupby(above) if k), default=0)
                if 'longest_below' in stats_to_extract:
                    stats_dict['longest_below'] = max((sum(1 for _ in g) for k, g in itertools.groupby(~above) if k), default=0)
            except Exception:
                if 'longest_above' in stats_to_extract:
                    stats_dict['longest_above'] = np.nan
                if 'longest_below' in stats_to_extract:
                    stats_dict['longest_below'] = np.nan

        # abs energy
        if 'abs_energy' in stats_to_extract:
            stats_dict['abs_energy'] = np.sum(values**2)

        # FFT амплитуды
        if 'fft_amp1' in stats_to_extract or 'fft_amp2' in stats_to_extract:
            try:
                fft = np.fft.rfft(values - np.mean(values))
                amplitudes = np.abs(fft)
                if 'fft_amp1' in stats_to_extract:
                    stats_dict['fft_amp1'] = amplitudes[1] if len(amplitudes) > 1 else np.nan
                if 'fft_amp2' in stats_to_extract:
                    stats_dict['fft_amp2'] = amplitudes[2] if len(amplitudes) > 2 else np.nan
            except Exception:
                stats_dict['fft_amp1'] = stats_dict['fft_amp2'] = np.nan

        # Mann-Kendall test
        if 'mk_trend' in stats_to_extract or 'mk_p' in stats_to_extract:
            try:
                mk_result = mk.original_test(values)
                if 'mk_trend' in stats_to_extract:
                    stats_dict['mk_trend'] = mk_result.trend
                if 'mk_p' in stats_to_extract:
                    stats_dict['mk_p'] = mk_result.p
            except Exception:
                if 'mk_trend' in stats_to_extract:
                    stats_dict['mk_trend'] = np.nan
                if 'mk_p' in stats_to_extract:
                    stats_dict['mk_p'] = np.nan

        # Theil-Sen slope
        if 'ts_slope' in stats_to_extract:
            try:
                ts_model = TheilSenRegressor()
                ts_model.fit(X_time, values)
                stats_dict['ts_slope'] = ts_model.coef_[0]
            except Exception:
                stats_dict['ts_slope'] = np.nan

        # Hurst exponent
        if 'hurst' in stats_to_extract:
            try:
                lags = range(2, 20)
                tau = [np.sqrt(np.std(np.subtract(values[lag:], values[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                stats_dict['hurst'] = poly[0] * 2.0
            except Exception:
                stats_dict['hurst'] = np.nan

        # autocorrelation lag-1
        if 'ac1' in stats_to_extract:
            try:
                if np.var(values) == 0:
                    stats_dict['ac1'] = 0
                else:
                    stats_dict['ac1'] = np.corrcoef(values[:-1], values[1:])[0, 1]
            except Exception:
                stats_dict['ac1'] = np.nan

        # STL trend slope
        if 'stl_slope' in stats_to_extract:
            try:
                stl = STL(pd.Series(values), period=max(3, len(values) // 2)).fit()
                trend = stl.trend
                stl_slope, _, _, _, _ = stats.linregress(np.arange(len(trend)), trend)
                stats_dict['stl_slope'] = stl_slope
            except Exception:
                stats_dict['stl_slope'] = np.nan

        return times, preds, stats_dict

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

    def effective_sample_size(self, data: np.ndarray, lag: int = 1) -> int:
        r = acf(data, nlags=lag, fft=False)[lag]
        n = len(data)
        return int(n * (1 - r) / (1 + r)) if (1 + r) != 0 else n

    def detect_stats_drift(self, stats_matrix, new_stats_matrix, stats_names, window_size=50, exploitation_dates=None, drift_threshold=2.0, ks_pvalue=0.05):
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

        if window_size != 50:
            window_size = self.effective_sample_size(stats_matrix.shape[0])
            window_size = np.clip(window_size, 50, 250)

        if len(stats_names) != stats_matrix.shape[1] or len(stats_names) != new_stats_matrix.shape[1]:
            raise ValueError("stats_names must match the number of columns in stats_matrix and new_stats_matrix.")
        if stats_matrix.shape[0] < window_size or new_stats_matrix.shape[0] < window_size * 0.4:
            logging.info("Insufficient data for drift detection")
            return {stat_name: {'drift_detected': False, 'stats': {}} for stat_name in stats_names}

        results = {}
        for col_idx, stat_name in enumerate(stats_names):


            historical_values = stats_matrix[:window_size, col_idx] if stats_matrix.shape[0] > window_size else stats_matrix[:, col_idx]
            new_values = new_stats_matrix[-window_size:, col_idx] if new_stats_matrix.shape[0] > window_size else new_stats_matrix[:, col_idx]

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
        if stats_matrix.shape[0] <256:
            logging.warning("Too few observations in general data, returning no anomalies")
            return {
                'anomalies': np.zeros(new_stats_matrix.shape[0], dtype=bool),
                'stats': {'n_anomalies': 0}
            }
        if new_stats_matrix.shape[0] < 5:
            logging.warning("Too few observations in new data, returning no anomalies")
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
