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
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
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

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get input dimension from checkpoint or config
        # We'll need to load data first to get this
        pass
    
    def load_test_data(self):
        """Load test dataset"""
        logger.info("Loading test dataset...")

        # Check if using tuplet dataset
        use_tuplet = self.config.dataset.get('use_tuplet', False)

        if use_tuplet:
            return self.load_tuplet_test_data()
        else:
            return self.load_dsl_test_data()

    def load_tuplet_test_data(self):
        """Load tuplet test dataset"""
        # Load tuplet dataset
        dataset_path = self.config.dataset.tuplet_path
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join('Keystroke Dynamics', dataset_path)

        X_A, X_B, labels, feature_names = self.preprocessor.load_tuplet_dataset(dataset_path)

        # Split data (same as training: 80% train, 10% val, 10% test)
        n_samples = len(labels)
        np.random.seed(42)  # Use same seed as training
        indices = np.random.permutation(n_samples)

        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)

        test_idx = indices[train_size + val_size:]

        # Get test data
        X_A_test, X_B_test, y_test = X_A[test_idx], X_B[test_idx], labels[test_idx]

        # Normalize features (use same scaler as training)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

        # Fit on all data (in real scenario, we'd save the scaler from training)
        scaler.fit(np.vstack([X_A, X_B]))

        X_A_test = scaler.transform(X_A_test)
        X_B_test = scaler.transform(X_B_test)

        # Convert to tensors
        X_A_test = torch.FloatTensor(X_A_test)
        X_B_test = torch.FloatTensor(X_B_test)
        y_test = torch.LongTensor(y_test)

        logger.info(f"Tuplet test data loaded: {len(y_test)} pairs")
        logger.info(f"Features: {X_A_test.shape[1]}")
        logger.info(f"Genuine: {torch.sum(y_test == 1).item()}, Impostor: {torch.sum(y_test == 0).item()}")

        # Store for later use
        self.use_tuplet = True
        self.X_A_test = X_A_test
        self.X_B_test = X_B_test
        self.y_test = y_test

        return X_A_test, y_test, None  # Return None for test_df (not needed for tuplet)

    def load_dsl_test_data(self):
        """Load DSL test dataset"""
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

        logger.info(f"DSL test data loaded: {X_test.shape}")

        self.use_tuplet = False

        return X_test, y_test, test_df
    
    def evaluate_verification_tuplet(self):
        """Evaluate verification performance for tuplet dataset"""
        logger.info("Evaluating verification performance (tuplet dataset)...")

        self.model.eval()

        genuine_scores = []
        impostor_scores = []

        with torch.no_grad():
            # Compute embeddings for all test samples
            embeddings_a = self.model(self.X_A_test.to(self.device))
            embeddings_b = self.model(self.X_B_test.to(self.device))

            # Compute cosine similarity
            similarities = torch.nn.functional.cosine_similarity(embeddings_a, embeddings_b)

            # Separate genuine and impostor scores
            for i in range(len(self.y_test)):
                score = similarities[i].item()
                if self.y_test[i] == 1:  # Genuine
                    genuine_scores.append(score)
                else:  # Impostor
                    impostor_scores.append(score)

        logger.info(f"Genuine scores: {len(genuine_scores)}, Impostor scores: {len(impostor_scores)}")

        # Compute EER
        from src.keystroke_verification import KeystrokeVerifier
        verifier = KeystrokeVerifier(self.model, self.config)
        eer, optimal_threshold = verifier.compute_eer(genuine_scores, impostor_scores)

        # Use configured threshold instead of EER-based optimal threshold
        configured_threshold = self.config.verification.get('threshold', 0.85)
        use_adaptive = self.config.verification.get('adaptive_threshold', False)

        # Choose threshold based on configuration
        threshold_to_use = optimal_threshold if use_adaptive else configured_threshold
        logger.info(f"Using threshold: {threshold_to_use} (adaptive={use_adaptive}, configured={configured_threshold}, optimal={optimal_threshold})")

        # Compute additional metrics at chosen threshold
        scores = np.array(genuine_scores + impostor_scores)
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        predictions = (scores >= threshold_to_use).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # False Accept Rate (FAR) and False Reject Rate (FRR)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Genuine Accept Rate (GAR) and True Reject Rate (TRR)
        gar = tp / (tp + fn) if (tp + fn) > 0 else 0
        trr = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Display metrics in terminal
        self.display_metrics_terminal(eer, threshold_to_use, accuracy, precision, recall, f1,
                                     far, frr, gar, trr, cm, genuine_scores, impostor_scores)

        # Plot ROC curve
        roc_auc = self.plot_roc_curve(genuine_scores, impostor_scores)

        return {
            'eer': eer,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'gar': gar,
            'trr': trr,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores
        }

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

            # Need at least 100 samples: 50 for enrollment + 50 for testing
            if len(subject_indices) < 100:
                continue

            # Split into enrollment and verification
            # Use 50 samples for enrollment (minimum required)
            enroll_indices = subject_indices[:50]
            verify_indices = subject_indices[50:100]

            # Enroll user
            enroll_samples = X_test[enroll_indices]
            self.verifier.enroll_user(subject, enroll_samples)
            
            # Genuine verification (test 25 samples from this user)
            for idx in verify_indices[:25]:
                result = self.verifier.verify_user(subject, X_test[idx])
                genuine_scores.append(result['confidence'])

            # Impostor verification (test samples from other subjects)
            impostor_subjects = [s for s in subjects if s != subject][:5]
            for imp_subject in impostor_subjects:
                imp_mask = test_df['subject'] == imp_subject
                imp_indices = np.where(imp_mask.values)[0][:5]  # 5 samples per impostor

                for idx in imp_indices:
                    result = self.verifier.verify_user(subject, X_test[idx])
                    impostor_scores.append(result['confidence'])
        
        # Compute EER
        eer, optimal_threshold = self.verifier.compute_eer(genuine_scores, impostor_scores)

        # Compute additional metrics at optimal threshold
        scores = np.array(genuine_scores + impostor_scores)
        labels = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
        predictions = (scores >= optimal_threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # False Accept Rate (FAR) and False Reject Rate (FRR)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Genuine Accept Rate (GAR) and True Reject Rate (TRR)
        gar = tp / (tp + fn) if (tp + fn) > 0 else 0
        trr = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Display metrics in terminal
        self.display_metrics_terminal(eer, optimal_threshold, accuracy, precision, recall, f1,
                                     far, frr, gar, trr, cm, genuine_scores, impostor_scores)

        # Plot ROC curve
        roc_auc = self.plot_roc_curve(genuine_scores, impostor_scores)

        return {
            'eer': eer,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'far': far,
            'frr': frr,
            'gar': gar,
            'trr': trr,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores
        }
    
    def display_metrics_terminal(self, eer, optimal_threshold, accuracy, precision, recall, f1,
                                 far, frr, gar, trr, cm, genuine_scores, impostor_scores):
        """Display comprehensive metrics in terminal"""

        print("\n" + "="*80)
        print("📊 KEYSTROKE DYNAMICS MODEL - PERFORMANCE METRICS")
        print("="*80)

        # Primary Metrics
        print("\n🎯 PRIMARY METRICS")
        print("-" * 80)
        print(f"  Equal Error Rate (EER)      : {eer:.4f} ({eer*100:.2f}%)")
        print(f"  Optimal Threshold           : {optimal_threshold:.4f}")
        print(f"  Accuracy                    : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision                   : {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall (Sensitivity/TPR)    : {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score                    : {f1:.4f}")

        # Biometric-Specific Metrics
        print("\n🔐 BIOMETRIC AUTHENTICATION METRICS")
        print("-" * 80)
        print(f"  False Accept Rate (FAR)     : {far:.4f} ({far*100:.2f}%)")
        print(f"  False Reject Rate (FRR)     : {frr:.4f} ({frr*100:.2f}%)")
        print(f"  Genuine Accept Rate (GAR)   : {gar:.4f} ({gar*100:.2f}%)")
        print(f"  True Reject Rate (TRR)      : {trr:.4f} ({trr*100:.2f}%)")

        # Confusion Matrix
        tn, fp, fn, tp = cm.ravel()
        print("\n📋 CONFUSION MATRIX")
        print("-" * 80)
        print(f"                    Predicted")
        print(f"                 Genuine  Impostor")
        print(f"  Actual Genuine    {tp:4d}     {fn:4d}      (TP: {tp}, FN: {fn})")
        print(f"         Impostor   {fp:4d}     {tn:4d}      (FP: {fp}, TN: {tn})")
        print(f"\n  True Positives (TP)  : {tp:4d} - Genuine users correctly verified")
        print(f"  True Negatives (TN)  : {tn:4d} - Impostors correctly rejected")
        print(f"  False Positives (FP) : {fp:4d} - Impostors incorrectly accepted ⚠️")
        print(f"  False Negatives (FN) : {fn:4d} - Genuine users incorrectly rejected ⚠️")

        # Score Statistics
        genuine_mean = np.mean(genuine_scores)
        genuine_std = np.std(genuine_scores)
        impostor_mean = np.mean(impostor_scores)
        impostor_std = np.std(impostor_scores)
        score_separation = genuine_mean - impostor_mean

        print("\n📈 SCORE STATISTICS")
        print("-" * 80)
        print(f"  Genuine Scores:")
        print(f"    Mean                      : {genuine_mean:.4f} ({genuine_mean*100:.2f}%)")
        print(f"    Std Dev                   : {genuine_std:.4f}")
        print(f"    Min                       : {np.min(genuine_scores):.4f}")
        print(f"    Max                       : {np.max(genuine_scores):.4f}")
        print(f"\n  Impostor Scores:")
        print(f"    Mean                      : {impostor_mean:.4f} ({impostor_mean*100:.2f}%)")
        print(f"    Std Dev                   : {impostor_std:.4f}")
        print(f"    Min                       : {np.min(impostor_scores):.4f}")
        print(f"    Max                       : {np.max(impostor_scores):.4f}")
        print(f"\n  Score Separation            : {score_separation:.4f} ({score_separation*100:.2f}%)")

        # Performance Targets
        print("\n🎯 PERFORMANCE TARGETS")
        print("-" * 80)
        eer_target = 0.05
        accuracy_target = 0.95
        f1_target = 0.95

        eer_status = "✅ PASS" if eer <= eer_target else "❌ FAIL"
        accuracy_status = "✅ PASS" if accuracy >= accuracy_target else "❌ FAIL"
        f1_status = "✅ PASS" if f1 >= f1_target else "❌ FAIL"

        print(f"  EER Target (<5%)            : {eer_status} (Current: {eer*100:.2f}%, Target: {eer_target*100:.2f}%)")
        print(f"  Accuracy Target (>95%)      : {accuracy_status} (Current: {accuracy*100:.2f}%, Target: {accuracy_target*100:.2f}%)")
        print(f"  F1 Score Target (>0.95)     : {f1_status} (Current: {f1:.4f}, Target: {f1_target:.4f})")

        # Sample Counts
        print("\n📊 TEST SAMPLES")
        print("-" * 80)
        print(f"  Genuine Samples Tested      : {len(genuine_scores)}")
        print(f"  Impostor Samples Tested     : {len(impostor_scores)}")
        print(f"  Total Samples               : {len(genuine_scores) + len(impostor_scores)}")

        print("\n" + "="*80)
        print()

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

        return roc_auc
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
        else:
            logger.warning("No checkpoint found, using untrained model")

        # Evaluate verification
        if hasattr(self, 'use_tuplet') and self.use_tuplet:
            verification_results = self.evaluate_verification_tuplet()
        else:
            verification_results = self.evaluate_verification(X_test, y_test, test_df)

        # Evaluate anomaly detection
        if hasattr(self, 'use_tuplet') and self.use_tuplet:
            # For tuplet, use X_A for anomaly detection
            anomaly_results = self.evaluate_anomaly_detection(self.X_A_test, self.y_test)
        else:
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

            # Primary Metrics
            f.write("PRIMARY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Equal Error Rate (EER)      : {verification_results['eer']:.4f} ({verification_results['eer']*100:.2f}%)\n")
            f.write(f"Optimal Threshold           : {verification_results['optimal_threshold']:.4f}\n")
            f.write(f"Accuracy                    : {verification_results['accuracy']:.4f} ({verification_results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision                   : {verification_results['precision']:.4f} ({verification_results['precision']*100:.2f}%)\n")
            f.write(f"Recall (Sensitivity/TPR)    : {verification_results['recall']:.4f} ({verification_results['recall']*100:.2f}%)\n")
            f.write(f"F1 Score                    : {verification_results['f1_score']:.4f}\n")
            f.write(f"ROC AUC                     : {verification_results['roc_auc']:.4f}\n\n")

            # Biometric Metrics
            f.write("BIOMETRIC AUTHENTICATION METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"False Accept Rate (FAR)     : {verification_results['far']:.4f} ({verification_results['far']*100:.2f}%)\n")
            f.write(f"False Reject Rate (FRR)     : {verification_results['frr']:.4f} ({verification_results['frr']*100:.2f}%)\n")
            f.write(f"Genuine Accept Rate (GAR)   : {verification_results['gar']:.4f} ({verification_results['gar']*100:.2f}%)\n")
            f.write(f"True Reject Rate (TRR)      : {verification_results['trr']:.4f} ({verification_results['trr']*100:.2f}%)\n\n")

            # Confusion Matrix
            cm = verification_results['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write(f"                    Predicted\n")
            f.write(f"                 Genuine  Impostor\n")
            f.write(f"  Actual Genuine    {tp:4d}     {fn:4d}      (TP: {tp}, FN: {fn})\n")
            f.write(f"         Impostor   {fp:4d}     {tn:4d}      (FP: {fp}, TN: {tn})\n\n")
            f.write(f"  True Positives (TP)  : {tp:4d} - Genuine users correctly verified\n")
            f.write(f"  True Negatives (TN)  : {tn:4d} - Impostors correctly rejected\n")
            f.write(f"  False Positives (FP) : {fp:4d} - Impostors incorrectly accepted\n")
            f.write(f"  False Negatives (FN) : {fn:4d} - Genuine users incorrectly rejected\n\n")

            # Score Statistics
            genuine_mean = np.mean(verification_results['genuine_scores'])
            genuine_std = np.std(verification_results['genuine_scores'])
            impostor_mean = np.mean(verification_results['impostor_scores'])
            impostor_std = np.std(verification_results['impostor_scores'])

            f.write("SCORE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Genuine Scores:\n")
            f.write(f"  Mean                      : {genuine_mean:.4f} ({genuine_mean*100:.2f}%)\n")
            f.write(f"  Std Dev                   : {genuine_std:.4f}\n")
            f.write(f"  Min                       : {np.min(verification_results['genuine_scores']):.4f}\n")
            f.write(f"  Max                       : {np.max(verification_results['genuine_scores']):.4f}\n\n")
            f.write(f"Impostor Scores:\n")
            f.write(f"  Mean                      : {impostor_mean:.4f} ({impostor_mean*100:.2f}%)\n")
            f.write(f"  Std Dev                   : {impostor_std:.4f}\n")
            f.write(f"  Min                       : {np.min(verification_results['impostor_scores']):.4f}\n")
            f.write(f"  Max                       : {np.max(verification_results['impostor_scores']):.4f}\n\n")
            f.write(f"Score Separation            : {genuine_mean - impostor_mean:.4f} ({(genuine_mean - impostor_mean)*100:.2f}%)\n\n")

            # Sample Counts
            f.write("TEST SAMPLES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Genuine Samples Tested      : {len(verification_results['genuine_scores'])}\n")
            f.write(f"Impostor Samples Tested     : {len(verification_results['impostor_scores'])}\n")
            f.write(f"Total Samples               : {len(verification_results['genuine_scores']) + len(verification_results['impostor_scores'])}\n\n")

            # Anomaly Detection
            f.write("ANOMALY DETECTION PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples               : {anomaly_results['total_samples']}\n")
            f.write(f"Anomalies Detected          : {anomaly_results['anomaly_count']}\n")
            f.write(f"Anomaly Rate                : {anomaly_results['anomaly_rate']*100:.2f}%\n")
            f.write(f"Mean Anomaly Score          : {anomaly_results['mean_score']:.4f}\n\n")

            # Performance Targets
            f.write("PERFORMANCE TARGETS\n")
            f.write("-" * 80 + "\n")
            eer_target = self.config.performance.eer_threshold
            accuracy_target = 0.95
            f1_target = 0.95

            eer_status = "PASS" if verification_results['eer'] <= eer_target else "FAIL"
            accuracy_status = "PASS" if verification_results['accuracy'] >= accuracy_target else "FAIL"
            f1_status = "PASS" if verification_results['f1_score'] >= f1_target else "FAIL"

            f.write(f"EER Target (<5%)            : {eer_status} (Current: {verification_results['eer']*100:.2f}%, Target: {eer_target*100:.2f}%)\n")
            f.write(f"Accuracy Target (>95%)      : {accuracy_status} (Current: {verification_results['accuracy']*100:.2f}%, Target: {accuracy_target*100:.2f}%)\n")
            f.write(f"F1 Score Target (>0.95)     : {f1_status} (Current: {verification_results['f1_score']:.4f}, Target: {f1_target:.4f})\n\n")

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

