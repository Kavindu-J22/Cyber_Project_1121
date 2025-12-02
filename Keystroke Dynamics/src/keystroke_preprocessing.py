"""
Keystroke Data Preprocessing Module
Handles loading, feature extraction, and preprocessing of keystroke timing data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from loguru import logger
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats


class KeystrokePreprocessor:
    """
    Preprocessor for keystroke dynamics data
    Extracts and normalizes timing features from keystroke sequences
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
        
        logger.info("KeystrokePreprocessor initialized")
    
    def load_dsl_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load DSL-StrongPasswordData dataset
        
        Args:
            file_path: Path to dataset file (.xls or .csv)
            
        Returns:
            DataFrame with keystroke timing data
        """
        logger.info(f"Loading DSL dataset from: {file_path}")
        
        try:
            if file_path.endswith('.xls'):
                # Try with xlrd
                df = pd.read_excel(file_path, engine='xlrd')
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
            logger.info(f"Subjects: {df['subject'].nunique() if 'subject' in df.columns else 'N/A'}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def extract_timing_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract timing features from DSL dataset
        
        Features include:
        - Hold times (H.key)
        - Keydown-keydown times (DD.key1.key2)
        - Keyup-keydown times (UD.key1.key2)
        
        Args:
            df: DataFrame with keystroke data
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("Extracting timing features...")
        
        # Identify feature columns
        hold_cols = [col for col in df.columns if col.startswith('H.')]
        dd_cols = [col for col in df.columns if col.startswith('DD.')]
        ud_cols = [col for col in df.columns if col.startswith('UD.')]
        
        feature_cols = []
        if self.config.features.hold_times:
            feature_cols.extend(hold_cols)
        if self.config.features.dd_times:
            feature_cols.extend(dd_cols)
        if self.config.features.ud_times:
            feature_cols.extend(ud_cols)
        
        self.feature_names = feature_cols
        
        # Extract features
        X = df[feature_cols].values
        
        # Extract labels (subject IDs)
        if 'subject' in df.columns:
            # Convert subject IDs to numeric labels
            subjects = df['subject'].unique()
            subject_to_id = {subj: idx for idx, subj in enumerate(subjects)}
            y = df['subject'].map(subject_to_id).values
        else:
            y = np.zeros(len(df))
        
        logger.info(f"Extracted {len(feature_cols)} timing features")
        logger.info(f"Feature types: {len(hold_cols)} hold, {len(dd_cols)} DD, {len(ud_cols)} UD")
        
        return X, y, feature_cols
    
    def compute_statistical_features(self, X: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Compute statistical features over sliding windows
        
        Args:
            X: Input features (n_samples, n_features)
            window_size: Size of sliding window
            
        Returns:
            Statistical features
        """
        if not self.config.features.use_statistics:
            return X
        
        logger.info("Computing statistical features...")
        
        stats_features = []
        stats_list = self.config.features.statistics
        
        for stat_name in stats_list:
            if stat_name == 'mean':
                stats_features.append(np.mean(X, axis=1, keepdims=True))
            elif stat_name == 'std':
                stats_features.append(np.std(X, axis=1, keepdims=True))
            elif stat_name == 'median':
                stats_features.append(np.median(X, axis=1, keepdims=True))
            elif stat_name == 'min':
                stats_features.append(np.min(X, axis=1, keepdims=True))
            elif stat_name == 'max':
                stats_features.append(np.max(X, axis=1, keepdims=True))
            elif stat_name == 'q25':
                stats_features.append(np.percentile(X, 25, axis=1, keepdims=True))
            elif stat_name == 'q75':
                stats_features.append(np.percentile(X, 75, axis=1, keepdims=True))
        
        # Concatenate original and statistical features
        if stats_features:
            stats_array = np.concatenate(stats_features, axis=1)
            X_enhanced = np.concatenate([X, stats_array], axis=1)
            logger.info(f"Added {stats_array.shape[1]} statistical features")
            return X_enhanced
        
        return X

    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using robust scaling

        Args:
            X: Input features
            fit: Whether to fit the scaler

        Returns:
            Normalized features
        """
        if fit:
            logger.info("Fitting scaler on training data...")
            X_normalized = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                logger.warning("Scaler not fitted, fitting now...")
                X_normalized = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                X_normalized = self.scaler.transform(X)

        return X_normalized

    def handle_outliers(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Handle outliers using z-score method

        Args:
            X: Input features
            threshold: Z-score threshold

        Returns:
            Features with outliers clipped
        """
        logger.info("Handling outliers...")

        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
        X_clean = np.where(z_scores > threshold,
                          np.nanmedian(X, axis=0),
                          X)

        outlier_count = np.sum(z_scores > threshold)
        logger.info(f"Clipped {outlier_count} outlier values")

        return X_clean

    def augment_data(self, X: np.ndarray, y: np.ndarray,
                     augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment keystroke data with noise and time warping

        Args:
            X: Input features
            y: Labels
            augmentation_factor: Number of augmented samples per original

        Returns:
            Augmented features and labels
        """
        if not self.config.training.use_augmentation:
            return X, y

        logger.info(f"Augmenting data with factor {augmentation_factor}...")

        X_aug_list = [X]
        y_aug_list = [y]

        noise_level = self.config.training.noise_level

        for _ in range(augmentation_factor - 1):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X + noise * np.std(X, axis=0)

            # Time warping (slight scaling)
            if self.config.training.time_warping:
                warp_factor = np.random.uniform(0.95, 1.05, (X.shape[0], 1))
                X_warped = X_noisy * warp_factor
            else:
                X_warped = X_noisy

            X_aug_list.append(X_warped)
            y_aug_list.append(y)

        X_augmented = np.vstack(X_aug_list)
        y_augmented = np.concatenate(y_aug_list)

        logger.info(f"Augmented dataset: {X.shape[0]} -> {X_augmented.shape[0]} samples")

        return X_augmented, y_augmented

    def preprocess_pipeline(self, df: pd.DataFrame,
                           fit: bool = False,
                           augment: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame
            fit: Whether to fit scalers
            augment: Whether to augment data

        Returns:
            Preprocessed features and labels as tensors
        """
        logger.info("Running preprocessing pipeline...")

        # Extract timing features
        X, y, feature_names = self.extract_timing_features(df)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Handle outliers
        X = self.handle_outliers(X)

        # Compute statistical features
        X = self.compute_statistical_features(X)

        # Normalize
        X = self.normalize_features(X, fit=fit)

        # Augment if requested
        if augment:
            X, y = self.augment_data(X, y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        logger.info(f"Preprocessing complete: {X_tensor.shape}")

        return X_tensor, y_tensor

    def split_by_subject(self, df: pd.DataFrame,
                        train_ratio: float = 0.6,
                        val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by subject for proper evaluation

        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Train, validation, and test DataFrames
        """
        logger.info("Splitting dataset by subject...")

        subjects = df['subject'].unique()
        n_subjects = len(subjects)

        # Shuffle subjects
        np.random.shuffle(subjects)

        # Split subjects
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)

        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]

        train_df = df[df['subject'].isin(train_subjects)]
        val_df = df[df['subject'].isin(val_subjects)]
        test_df = df[df['subject'].isin(test_subjects)]

        logger.info(f"Split: Train={len(train_df)} ({len(train_subjects)} subjects), "
                   f"Val={len(val_df)} ({len(val_subjects)} subjects), "
                   f"Test={len(test_df)} ({len(test_subjects)} subjects)")

        return train_df, val_df, test_df

