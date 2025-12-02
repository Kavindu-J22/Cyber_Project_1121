"""
Testing Script for Mouse Movement Analysis Model
Evaluates model performance on Balabit test dataset
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config
from src.mouse_preprocessing import MousePreprocessor
from src.mouse_embedding import MouseEmbeddingModel
from src.mouse_verification import MouseVerifier
from src.anomaly_detection import AnomalyDetector


class MouseTester:
    """Tester for mouse movement dynamics model"""
    
    def __init__(self, config):
        """Initialize tester"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = MousePreprocessor(config)
        self.model = None
        self.verifier = None
        
        # Results
        self.results = {
            'genuine_scores': [],
            'impostor_scores': [],
            'genuine_labels': [],
            'impostor_labels': []
        }
    
    def load_model(self, checkpoint_path: str):
        """Load trained model"""
        logger.info(f"Loading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get input dimension from checkpoint or infer from data
        # For now, we'll load data first to get input_dim
        return checkpoint
    
    def load_test_data(self):
        """Load test dataset"""
        logger.info("Loading test dataset...")
        
        # Load test data
        test_path = self.config.dataset.test_files
        labels_path = self.config.dataset.labels_file
        
        dataset = self.preprocessor.load_balabit_dataset(test_path, labels_path, is_training=False)
        
        # Load labels
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            logger.info(f"Loaded {len(labels_df)} test labels")
        else:
            labels_df = None
            logger.warning("No labels file found")
        
        return dataset, labels_df
    
    def evaluate_verification(self, dataset, labels_df):
        """Evaluate verification performance"""
        logger.info("Evaluating verification performance...")
        
        all_scores = []
        all_labels = []
        
        # Process each user
        for user_id, session_files in tqdm(dataset.items(), desc="Processing users"):
            # Enroll user with first few sessions
            enroll_sessions = session_files[:min(3, len(session_files))]
            test_sessions = session_files[min(3, len(session_files)):]
            
            if len(test_sessions) == 0:
                continue
            
            # Extract enrollment features
            enroll_features = []
            for session_file in enroll_sessions:
                features = self.preprocessor.extract_session_features(session_file)
                if features is not None:
                    enroll_features.append(features)
            
            if len(enroll_features) == 0:
                continue
            
            # Concatenate and normalize
            X_enroll = np.vstack(enroll_features)
            X_enroll = self.preprocessor.normalize_features(X_enroll, fit=False)
            X_enroll_tensor = torch.FloatTensor(X_enroll)
            
            # Enroll user
            try:
                self.verifier.enroll_user(user_id, X_enroll_tensor)
            except Exception as e:
                logger.error(f"Failed to enroll user {user_id}: {e}")
                continue
            
            # Test on remaining sessions
            for session_file in test_sessions:
                features = self.preprocessor.extract_session_features(session_file)
                
                if features is None:
                    continue
                
                # Normalize
                X_test = self.preprocessor.normalize_features(features, fit=False)
                X_test_tensor = torch.FloatTensor(X_test)
                
                # Verify each sample
                for i in range(len(X_test_tensor)):
                    result = self.verifier.verify_user(user_id, X_test_tensor[i])
                    
                    # Get session name
                    session_name = os.path.basename(session_file)
                    
                    # Check if this is genuine or impostor
                    is_genuine = 1  # Default to genuine (same user)
                    
                    if labels_df is not None:
                        label_row = labels_df[labels_df['filename'] == session_name]
                        if not label_row.empty:
                            is_genuine = 1 - int(label_row.iloc[0]['is_illegal'])  # 0 if illegal, 1 if legal
                    
                    all_scores.append(result['confidence'])
                    all_labels.append(is_genuine)
        
        return np.array(all_scores), np.array(all_labels)
    
    def calculate_metrics(self, scores, labels, threshold=None):
        """Calculate performance metrics"""
        if threshold is None:
            threshold = self.config.verification.threshold
        
        predictions = (scores >= threshold).astype(int)
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # Calculate AUC if we have both classes
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, scores)
        else:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': threshold
        }
        
        return metrics

    def calculate_eer(self, scores, labels):
        """Calculate Equal Error Rate"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        # Find where FPR and FNR are closest
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return eer, eer_threshold

    def plot_roc_curve(self, scores, labels):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Mouse Movement Analysis')
        plt.legend()
        plt.grid(True)

        os.makedirs(self.config.paths.logs_dir, exist_ok=True)
        plot_path = os.path.join(self.config.paths.logs_dir, 'roc_curve.png')
        plt.savefig(plot_path)
        logger.info(f"ROC curve saved: {plot_path}")
        plt.close()

    def plot_score_distribution(self, scores, labels):
        """Plot score distribution"""
        genuine_scores = scores[labels == 1]
        impostor_scores = scores[labels == 0]

        plt.figure(figsize=(10, 6))
        plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='green')
        plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red')
        plt.axvline(self.config.verification.threshold, color='black', linestyle='--',
                   label=f'Threshold ({self.config.verification.threshold})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution - Mouse Movement Analysis')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.config.paths.logs_dir, 'score_distribution.png')
        plt.savefig(plot_path)
        logger.info(f"Score distribution saved: {plot_path}")
        plt.close()

    def run_test(self):
        """Run complete test pipeline"""
        logger.info("Starting test pipeline...")

        # Load test data
        dataset, labels_df = self.load_test_data()

        # Load model checkpoint
        checkpoint_path = os.path.join(self.config.paths.checkpoint_dir, 'best_model.pth')
        checkpoint = self.load_model(checkpoint_path)

        # We need to get input_dim from preprocessor
        # Extract features from first session to get dimension
        first_user = list(dataset.keys())[0]
        first_session = dataset[first_user][0]
        sample_features = self.preprocessor.extract_session_features(first_session)

        if sample_features is None:
            raise ValueError("Failed to extract features from sample session")

        input_dim = sample_features.shape[1]
        logger.info(f"Input dimension: {input_dim}")

        # Build model
        self.model = MouseEmbeddingModel(input_dim, self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully")

        # Initialize verifier
        self.verifier = MouseVerifier(self.model, self.config)

        # Evaluate
        scores, labels = self.evaluate_verification(dataset, labels_df)

        logger.info(f"Collected {len(scores)} verification samples")
        logger.info(f"Genuine samples: {np.sum(labels == 1)}")
        logger.info(f"Impostor samples: {np.sum(labels == 0)}")

        # Calculate metrics
        metrics = self.calculate_metrics(scores, labels)

        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"AUC:       {metrics['auc']:.4f}")

        # Calculate EER
        if len(np.unique(labels)) > 1:
            eer, eer_threshold = self.calculate_eer(scores, labels)
            logger.info(f"EER:       {eer:.4f} ({eer*100:.2f}%)")
            logger.info(f"EER Threshold: {eer_threshold:.4f}")

            # Plot ROC curve
            self.plot_roc_curve(scores, labels)

            # Plot score distribution
            self.plot_score_distribution(scores, labels)

        logger.info("=" * 60)

        # Save results
        results_path = os.path.join(self.config.paths.logs_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write("Mouse Movement Analysis - Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"AUC:       {metrics['auc']:.4f}\n")
            if len(np.unique(labels)) > 1:
                f.write(f"EER:       {eer:.4f} ({eer*100:.2f}%)\n")
                f.write(f"EER Threshold: {eer_threshold:.4f}\n")

        logger.info(f"Results saved to: {results_path}")

        return metrics


def main():
    """Main testing function"""
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger.add("logs/testing_{time}.log", rotation="100 MB")

    # Load configuration
    config = load_config('config.yaml')

    # Create tester
    tester = MouseTester(config)

    # Run tests
    tester.run_test()

    logger.info("Testing script completed successfully!")


if __name__ == '__main__':
    main()

