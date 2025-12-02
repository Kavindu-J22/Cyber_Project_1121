"""
Testing Script for Keystroke Dynamics Model
Evaluates model performance and computes metrics
"""

import os
import sys
import torch
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config_loader import load_config
from src.keystroke_preprocessing import KeystrokePreprocessor
from src.keystroke_embedding import KeystrokeEmbeddingModel
from src.keystroke_verification import KeystrokeVerifier
from src.anomaly_detection import AnomalyDetector


class KeystrokeTester:
    """Tester for keystroke dynamics model"""
    
    def __init__(self, config, model_path):
        """Initialize tester"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = KeystrokePreprocessor(config)
        self.model = None
        self.verifier = None
        self.anomaly_detector = AnomalyDetector(config)
        
        # Load model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get input dimension from checkpoint or config
        # We'll need to load data first to get this
        pass
    
    def load_test_data(self):
        """Load test dataset"""
        logger.info("Loading test dataset...")
        
        # Load DSL dataset
        dataset_path = os.path.join('Dataset', 'DSL-StrongPasswordData-Original_Dataset.xls')
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join('Keystroke Dynamics', 'Dataset', 
                                       'DSL-StrongPasswordData-Original_Dataset.xls')
        
        df = self.preprocessor.load_dsl_dataset(dataset_path)
        
        # Split by subject
        train_df, val_df, test_df = self.preprocessor.split_by_subject(
            df,
            train_ratio=self.config.training.train_ratio,
            val_ratio=self.config.training.val_ratio
        )
        
        # Preprocess test data
        X_test, y_test = self.preprocessor.preprocess_pipeline(
            test_df, fit=False, augment=False
        )
        
        logger.info(f"Test data loaded: {X_test.shape}")
        
        return X_test, y_test, test_df
    
    def evaluate_verification(self, X_test, y_test, test_df):
        """Evaluate verification performance"""
        logger.info("Evaluating verification performance...")
        
        # Initialize verifier
        self.verifier = KeystrokeVerifier(self.model, self.config)
        
        # Get unique subjects
        subjects = test_df['subject'].unique()
        
        genuine_scores = []
        impostor_scores = []
        
        # For each subject
        for subject in subjects[:10]:  # Test on first 10 subjects
            # Get subject's data
            subject_mask = test_df['subject'] == subject
            subject_indices = np.where(subject_mask.values)[0]
            
            if len(subject_indices) < 20:
                continue
            
            # Split into enrollment and verification
            enroll_indices = subject_indices[:10]
            verify_indices = subject_indices[10:20]
            
            # Enroll user
            enroll_samples = X_test[enroll_indices]
            self.verifier.enroll_user(subject, enroll_samples)
            
            # Genuine verification
            for idx in verify_indices:
                result = self.verifier.verify_user(subject, X_test[idx])
                genuine_scores.append(result['confidence'])
            
            # Impostor verification (other subjects)
            impostor_subjects = [s for s in subjects if s != subject][:5]
            for imp_subject in impostor_subjects:
                imp_mask = test_df['subject'] == imp_subject
                imp_indices = np.where(imp_mask.values)[0][:2]
                
                for idx in imp_indices:
                    result = self.verifier.verify_user(subject, X_test[idx])
                    impostor_scores.append(result['confidence'])
        
        # Compute EER
        eer, optimal_threshold = self.verifier.compute_eer(genuine_scores, impostor_scores)
        
        logger.info(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
        logger.info(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        # Plot ROC curve
        self.plot_roc_curve(genuine_scores, impostor_scores)
        
        return {
            'eer': eer,
            'optimal_threshold': optimal_threshold,
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores
        }
    
    def plot_roc_curve(self, genuine_scores, impostor_scores):
        """Plot ROC curve"""
        # Combine scores and labels
        scores = np.array(genuine_scores + impostor_scores)
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plot_path = os.path.join(self.config.paths.logs_dir, 'roc_curve.png')
        plt.savefig(plot_path)
        logger.info(f"ROC curve saved: {plot_path}")
        plt.close()

    def evaluate_anomaly_detection(self, X_test, y_test):
        """Evaluate anomaly detection"""
        logger.info("Evaluating anomaly detection...")

        # Train anomaly detector on normal samples
        normal_samples = X_test[:100]  # Use first 100 as normal
        self.anomaly_detector.fit(normal_samples.numpy())

        # Test on remaining samples
        test_samples = X_test[100:200]

        # Generate anomaly report
        report = self.anomaly_detector.get_anomaly_report(test_samples.numpy())

        logger.info(f"Anomaly Detection Report: {report}")

        return report

    def run_comprehensive_test(self):
        """Run comprehensive testing"""
        logger.info("Running comprehensive tests...")

        # Load test data
        X_test, y_test, test_df = self.load_test_data()

        # Initialize model
        input_dim = X_test.shape[1]
        self.model = KeystrokeEmbeddingModel(input_dim, self.config)
        self.model = self.model.to(self.device)

        # Load checkpoint
        checkpoint_path = os.path.join(self.config.paths.checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
        else:
            logger.warning("No checkpoint found, using untrained model")

        # Evaluate verification
        verification_results = self.evaluate_verification(X_test, y_test, test_df)

        # Evaluate anomaly detection
        anomaly_results = self.evaluate_anomaly_detection(X_test, y_test)

        # Generate report
        self.generate_test_report(verification_results, anomaly_results)

        logger.info("Comprehensive testing completed!")

    def generate_test_report(self, verification_results, anomaly_results):
        """Generate comprehensive test report"""
        report_path = os.path.join(self.config.paths.logs_dir, 'test_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KEYSTROKE DYNAMICS - TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("VERIFICATION PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Equal Error Rate (EER): {verification_results['eer']:.4f} ({verification_results['eer']*100:.2f}%)\n")
            f.write(f"Optimal Threshold: {verification_results['optimal_threshold']:.4f}\n")
            f.write(f"Genuine Samples: {len(verification_results['genuine_scores'])}\n")
            f.write(f"Impostor Samples: {len(verification_results['impostor_scores'])}\n")
            f.write(f"Mean Genuine Score: {np.mean(verification_results['genuine_scores']):.4f}\n")
            f.write(f"Mean Impostor Score: {np.mean(verification_results['impostor_scores']):.4f}\n\n")

            f.write("ANOMALY DETECTION PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {anomaly_results['total_samples']}\n")
            f.write(f"Anomalies Detected: {anomaly_results['anomaly_count']}\n")
            f.write(f"Anomaly Rate: {anomaly_results['anomaly_rate']*100:.2f}%\n")
            f.write(f"Mean Anomaly Score: {anomaly_results['mean_score']:.4f}\n\n")

            f.write("PERFORMANCE TARGETS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Target EER: {self.config.performance.eer_threshold*100:.2f}%\n")
            f.write(f"Achieved EER: {verification_results['eer']*100:.2f}%\n")
            f.write(f"Target Met: {'YES' if verification_results['eer'] <= self.config.performance.eer_threshold else 'NO'}\n\n")

            f.write("=" * 80 + "\n")

        logger.info(f"Test report saved: {report_path}")


def main():
    """Main testing function"""
    # Setup logging
    logger.add("logs/testing_{time}.log", rotation="100 MB")

    # Load configuration
    config = load_config('config.yaml')

    # Create tester
    model_path = os.path.join(config.paths.checkpoint_dir, 'best_model.pth')
    tester = KeystrokeTester(config, model_path)

    # Run tests
    tester.run_comprehensive_test()

    logger.info("Testing script completed successfully!")


if __name__ == '__main__':
    main()

