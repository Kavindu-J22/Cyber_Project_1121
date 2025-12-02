"""
Mouse Movement Data Preprocessing Module
Handles loading, feature extraction, and preprocessing of mouse dynamics data
from the Balabit Mouse Dynamics Challenge dataset
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from loguru import logger
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from scipy.spatial.distance import euclidean
import os
import glob


class MousePreprocessor:
    """
    Preprocessor for mouse movement dynamics data
    Extracts comprehensive behavioral features from cursor trajectories,
    click dynamics, and scrolling patterns
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.scaler = RobustScaler()  # Robust to outliers
        self.feature_names = []
        self.is_fitted = False
        
        logger.info("MousePreprocessor initialized")
    
    def load_balabit_dataset(self, data_path: str, labels_path: str = None, 
                            is_training: bool = True) -> Dict[str, List[str]]:
        """
        Load Balabit Mouse Dynamics Challenge dataset
        
        Args:
            data_path: Path to training_files or test_files directory
            labels_path: Path to labels CSV file (for test data)
            is_training: Whether loading training or test data
            
        Returns:
            Dictionary mapping user IDs to list of session file paths
        """
        logger.info(f"Loading Balabit dataset from: {data_path}")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Get all user directories
        user_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        dataset = {}
        for user_dir in user_dirs:
            user_id = user_dir.name
            # Get all session files for this user
            session_files = list(user_dir.glob('session_*'))
            dataset[user_id] = [str(f) for f in session_files]
        
        logger.info(f"Loaded {len(dataset)} users with {sum(len(v) for v in dataset.values())} sessions")
        
        # Load labels if provided
        self.labels = None
        if labels_path and Path(labels_path).exists():
            self.labels = pd.read_csv(labels_path)
            logger.info(f"Loaded {len(self.labels)} labels")
        
        return dataset
    
    def load_session_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single session file
        
        Args:
            file_path: Path to session file
            
        Returns:
            DataFrame with mouse event data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Rename columns for consistency
            df.columns = ['record_timestamp', 'client_timestamp', 'button', 'state', 'x', 'y']
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading session file {file_path}: {e}")
            return None
    
    def extract_session_features(self, file_path: str) -> Optional[np.ndarray]:
        """
        Extract comprehensive features from a single session
        
        Args:
            file_path: Path to session file
            
        Returns:
            Feature vector or None if extraction fails
        """
        df = self.load_session_file(file_path)
        
        if df is None or len(df) < self.config.features.min_events:
            return None
        
        # Use sliding window to extract multiple feature vectors
        window_size = self.config.features.window_size
        stride = self.config.features.window_stride if self.config.features.sliding_window else window_size
        
        features_list = []
        
        for start_idx in range(0, len(df) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_df = df.iloc[start_idx:end_idx]
            
            features = self._extract_window_features(window_df)
            if features is not None:
                features_list.append(features)
        
        if len(features_list) == 0:
            return None
        
        return np.array(features_list)
    
    def _extract_window_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract features from a window of mouse events
        
        Args:
            df: DataFrame with mouse events in window
            
        Returns:
            Feature vector
        """
        features = []
        feature_names = []
        
        # Extract movement features
        if self.config.features.velocity_features:
            vel_features, vel_names = self._extract_velocity_features(df)
            features.extend(vel_features)
            feature_names.extend(vel_names)
        
        if self.config.features.acceleration_features:
            acc_features, acc_names = self._extract_acceleration_features(df)
            features.extend(acc_features)
            feature_names.extend(acc_names)

        if self.config.features.curvature_features:
            curv_features, curv_names = self._extract_curvature_features(df)
            features.extend(curv_features)
            feature_names.extend(curv_names)

        if self.config.features.jerk_features:
            jerk_features, jerk_names = self._extract_jerk_features(df)
            features.extend(jerk_features)
            feature_names.extend(jerk_names)

        # Extract click features
        if self.config.features.click_features:
            click_features, click_names = self._extract_click_features(df)
            features.extend(click_features)
            feature_names.extend(click_names)

        # Extract trajectory features
        if self.config.features.trajectory_length:
            traj_features, traj_names = self._extract_trajectory_features(df)
            features.extend(traj_features)
            feature_names.extend(traj_names)

        # Extract temporal features
        if self.config.features.pause_duration:
            temp_features, temp_names = self._extract_temporal_features(df)
            features.extend(temp_features)
            feature_names.extend(temp_names)

        # Store feature names on first extraction
        if len(self.feature_names) == 0:
            self.feature_names = feature_names

        return np.array(features)

    def _extract_velocity_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract velocity-based features"""
        features = []
        names = []

        # Calculate distances and time differences
        x_diff = np.diff(df['x'].values)
        y_diff = np.diff(df['y'].values)
        distances = np.sqrt(x_diff**2 + y_diff**2)

        time_diff = np.diff(df['client_timestamp'].values)
        time_diff = np.maximum(time_diff, 1e-6)  # Avoid division by zero

        # Velocity (pixels per second)
        velocities = distances / time_diff

        # Statistical features
        if self.config.features.use_statistics:
            for stat_name in self.config.features.statistics:
                if stat_name == 'mean':
                    features.append(np.mean(velocities))
                    names.append('velocity_mean')
                elif stat_name == 'std':
                    features.append(np.std(velocities))
                    names.append('velocity_std')
                elif stat_name == 'median':
                    features.append(np.median(velocities))
                    names.append('velocity_median')
                elif stat_name == 'min':
                    features.append(np.min(velocities))
                    names.append('velocity_min')
                elif stat_name == 'max':
                    features.append(np.max(velocities))
                    names.append('velocity_max')
                elif stat_name == 'q25':
                    features.append(np.percentile(velocities, 25))
                    names.append('velocity_q25')
                elif stat_name == 'q75':
                    features.append(np.percentile(velocities, 75))
                    names.append('velocity_q75')
                elif stat_name == 'skew':
                    features.append(stats.skew(velocities))
                    names.append('velocity_skew')
                elif stat_name == 'kurtosis':
                    features.append(stats.kurtosis(velocities))
                    names.append('velocity_kurtosis')

        # Velocity in X and Y directions
        vx = x_diff / time_diff
        vy = y_diff / time_diff

        features.extend([np.mean(vx), np.std(vx), np.mean(vy), np.std(vy)])
        names.extend(['vx_mean', 'vx_std', 'vy_mean', 'vy_std'])

        return features, names

    def _extract_acceleration_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract acceleration-based features"""
        features = []
        names = []

        # Calculate velocity first
        x_diff = np.diff(df['x'].values)
        y_diff = np.diff(df['y'].values)
        time_diff = np.diff(df['client_timestamp'].values)
        time_diff = np.maximum(time_diff, 1e-6)

        velocities = np.sqrt(x_diff**2 + y_diff**2) / time_diff

        # Calculate acceleration
        if len(velocities) > 1:
            vel_diff = np.diff(velocities)
            time_diff2 = time_diff[1:]
            accelerations = vel_diff / np.maximum(time_diff2, 1e-6)

            # Statistical features
            if self.config.features.use_statistics:
                for stat_name in self.config.features.statistics:
                    if stat_name == 'mean':
                        features.append(np.mean(accelerations))
                        names.append('acceleration_mean')
                    elif stat_name == 'std':
                        features.append(np.std(accelerations))
                        names.append('acceleration_std')
                    elif stat_name == 'median':
                        features.append(np.median(accelerations))
                        names.append('acceleration_median')
                    elif stat_name == 'max':
                        features.append(np.max(np.abs(accelerations)))
                        names.append('acceleration_max_abs')
        else:
            # Not enough data
            num_stats = len(self.config.features.statistics) if self.config.features.use_statistics else 0
            features.extend([0.0] * num_stats)
            names.extend([f'acceleration_{s}' for s in self.config.features.statistics[:num_stats]])

        return features, names

    def _extract_curvature_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract curvature and angular variation features"""
        features = []
        names = []

        x = df['x'].values
        y = df['y'].values

        if len(x) < 3:
            return [0.0, 0.0, 0.0], ['curvature_mean', 'curvature_std', 'angle_change_mean']

        # Calculate angles between consecutive movements
        angles = []
        for i in range(len(x) - 2):
            dx1, dy1 = x[i+1] - x[i], y[i+1] - y[i]
            dx2, dy2 = x[i+2] - x[i+1], y[i+2] - y[i+1]

            # Angle between vectors
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            angle_change = np.abs(angle2 - angle1)

            # Normalize to [0, pi]
            if angle_change > np.pi:
                angle_change = 2 * np.pi - angle_change

            angles.append(angle_change)

        angles = np.array(angles)

        # Curvature approximation
        curvatures = angles / np.maximum(np.sqrt((x[1:-1] - x[:-2])**2 + (y[1:-1] - y[:-2])**2), 1e-6)

        features.extend([
            np.mean(curvatures),
            np.std(curvatures),
            np.mean(angles)
        ])
        names.extend(['curvature_mean', 'curvature_std', 'angle_change_mean'])

        return features, names

    def _extract_jerk_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract jerk (rate of change of acceleration) features"""
        features = []
        names = []

        # Calculate velocity and acceleration first
        x_diff = np.diff(df['x'].values)
        y_diff = np.diff(df['y'].values)
        time_diff = np.diff(df['client_timestamp'].values)
        time_diff = np.maximum(time_diff, 1e-6)

        velocities = np.sqrt(x_diff**2 + y_diff**2) / time_diff

        if len(velocities) > 2:
            vel_diff = np.diff(velocities)
            time_diff2 = time_diff[1:]
            accelerations = vel_diff / np.maximum(time_diff2, 1e-6)

            # Calculate jerk
            acc_diff = np.diff(accelerations)
            time_diff3 = time_diff2[1:]
            jerks = acc_diff / np.maximum(time_diff3, 1e-6)

            features.extend([
                np.mean(np.abs(jerks)),
                np.std(jerks)
            ])
            names.extend(['jerk_mean_abs', 'jerk_std'])
        else:
            features.extend([0.0, 0.0])
            names.extend(['jerk_mean_abs', 'jerk_std'])

        return features, names

    def _extract_click_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract click dynamics features"""
        features = []
        names = []

        # Count different button states
        button_counts = df['button'].value_counts()
        total_events = len(df)

        # Click frequency
        click_events = df[df['button'] != 'NoButton']
        click_ratio = len(click_events) / total_events if total_events > 0 else 0

        features.append(click_ratio)
        names.append('click_ratio')

        # Click hold duration (if available)
        if self.config.features.click_hold_duration:
            hold_durations = []

            i = 0
            while i < len(df) - 1:
                if df.iloc[i]['button'] != 'NoButton':
                    # Found a click start
                    start_time = df.iloc[i]['client_timestamp']
                    j = i + 1

                    # Find when button is released
                    while j < len(df) and df.iloc[j]['button'] != 'NoButton':
                        j += 1

                    if j < len(df):
                        end_time = df.iloc[j]['client_timestamp']
                        hold_durations.append(end_time - start_time)

                    i = j
                else:
                    i += 1

            if len(hold_durations) > 0:
                features.extend([
                    np.mean(hold_durations),
                    np.std(hold_durations)
                ])
                names.extend(['click_hold_mean', 'click_hold_std'])
            else:
                features.extend([0.0, 0.0])
                names.extend(['click_hold_mean', 'click_hold_std'])

        return features, names

    def _extract_trajectory_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract trajectory-based features"""
        features = []
        names = []

        x = df['x'].values
        y = df['y'].values

        # Total path length
        path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

        # Direct distance (start to end)
        direct_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

        # Trajectory efficiency (straightness)
        efficiency = direct_distance / path_length if path_length > 0 else 0

        features.extend([
            path_length,
            direct_distance,
            efficiency
        ])
        names.extend(['path_length', 'direct_distance', 'trajectory_efficiency'])

        return features, names

    def _extract_temporal_features(self, df: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Extract temporal pattern features"""
        features = []
        names = []

        # Time between movements
        time_diffs = np.diff(df['client_timestamp'].values)

        # Pause detection (movements with large time gaps)
        pause_threshold = np.median(time_diffs) * 3 if len(time_diffs) > 0 else 0.1
        pauses = time_diffs[time_diffs > pause_threshold]

        features.extend([
            np.mean(time_diffs),
            np.std(time_diffs),
            len(pauses),
            np.mean(pauses) if len(pauses) > 0 else 0
        ])
        names.extend(['time_diff_mean', 'time_diff_std', 'num_pauses', 'pause_duration_mean'])

        return features, names

    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using robust scaling

        Args:
            X: Feature matrix
            fit: Whether to fit the scaler

        Returns:
            Normalized features
        """
        if fit:
            self.scaler.fit(X)
            self.is_fitted = True
            logger.info("Feature scaler fitted")

        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call with fit=True first.")

        X_normalized = self.scaler.transform(X)

        return X_normalized

    def augment_features(self, X: np.ndarray) -> np.ndarray:
        """
        Augment features with noise for training robustness

        Args:
            X: Feature matrix

        Returns:
            Augmented features
        """
        if not self.config.training.use_augmentation:
            return X

        noise_level = self.config.training.noise_level
        noise = np.random.normal(0, noise_level, X.shape)

        X_augmented = X + noise

        return X_augmented

    def split_by_user(self, dataset: Dict[str, List[str]],
                     train_ratio: float = 0.6,
                     val_ratio: float = 0.2) -> Tuple[Dict, Dict, Dict]:
        """
        Split dataset by users for training, validation, and testing

        Args:
            dataset: Dictionary mapping user IDs to session files
            train_ratio: Ratio of users for training
            val_ratio: Ratio of users for validation

        Returns:
            Tuple of (train_dict, val_dict, test_dict)
        """
        users = list(dataset.keys())
        np.random.shuffle(users)

        n_users = len(users)
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)

        train_users = users[:n_train]
        val_users = users[n_train:n_train + n_val]
        test_users = users[n_train + n_val:]

        train_dict = {u: dataset[u] for u in train_users}
        val_dict = {u: dataset[u] for u in val_users}
        test_dict = {u: dataset[u] for u in test_users}

        logger.info(f"Split: {len(train_users)} train, {len(val_users)} val, {len(test_users)} test users")

        return train_dict, val_dict, test_dict

    def preprocess_pipeline(self, dataset: Dict[str, List[str]],
                           fit: bool = False,
                           augment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline

        Args:
            dataset: Dictionary mapping user IDs to session files
            fit: Whether to fit the scaler
            augment: Whether to augment data

        Returns:
            Tuple of (features_tensor, labels_tensor)
        """
        logger.info("Running preprocessing pipeline...")

        all_features = []
        all_labels = []

        for user_idx, (user_id, session_files) in enumerate(dataset.items()):
            logger.debug(f"Processing user {user_id} ({user_idx+1}/{len(dataset)})")

            for session_file in session_files:
                features = self.extract_session_features(session_file)

                if features is not None:
                    # features is (n_windows, n_features)
                    all_features.append(features)
                    all_labels.extend([user_idx] * len(features))

        if len(all_features) == 0:
            raise ValueError("No features extracted from dataset")

        # Concatenate all features
        X = np.vstack(all_features)
        y = np.array(all_labels)

        logger.info(f"Extracted features: {X.shape}, Labels: {y.shape}")

        # Normalize
        X = self.normalize_features(X, fit=fit)

        # Augment if requested
        if augment:
            X = self.augment_features(X)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        return X_tensor, y_tensor

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using random forest

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_

        feature_importance = {
            name: importance
            for name, importance in zip(self.feature_names, importances)
        }

        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(),
                                        key=lambda x: x[1],
                                        reverse=True))

        return feature_importance


