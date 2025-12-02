"""
Training Script for Voiceprint Analysis
Trains and evaluates speaker verification system on VoxCeleb dataset
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Tuple
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_config
from src.speaker_embedding import SpeakerEmbeddingModel
from src.audio_preprocessing import AudioPreprocessor
from src.speaker_verification import SpeakerVerificationEngine


class VoiceprintTrainer:
    """Trainer for voiceprint analysis system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.embedding_model = SpeakerEmbeddingModel(self.config)
        self.audio_preprocessor = AudioPreprocessor(self.config)
        
        # Dataset paths
        self.dataset_base = Path(self.config.get('dataset.base_path'))
        self.dev_audio_path = self.dataset_base / self.config.get('dataset.dev_audio')
        self.test_audio_path = self.dataset_base / self.config.get('dataset.test_audio')
        self.tuplets_csv = self.dataset_base / self.config.get('dataset.tuplets_csv')
        self.veri_test = self.dataset_base / self.config.get('dataset.veri_test')
        
        print(f"Dataset base path: {self.dataset_base}")
        print(f"Dev audio path: {self.dev_audio_path}")
        print(f"Test audio path: {self.test_audio_path}")
    
    def load_verification_pairs(self, pairs_file: Path) -> List[Tuple[int, str, str]]:
        """
        Load verification pairs from file
        
        Args:
            pairs_file: Path to verification pairs file
            
        Returns:
            List of (label, audio1_path, audio2_path) tuples
        """
        pairs = []
        
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    audio1 = parts[1]
                    audio2 = parts[2]
                    pairs.append((label, audio1, audio2))
        
        return pairs
    
    def compute_eer(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER)
        
        Args:
            genuine_scores: Similarity scores for genuine pairs
            impostor_scores: Similarity scores for impostor pairs
            
        Returns:
            Tuple of (EER, threshold)
        """
        # Combine scores and labels
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Sort by scores
        sorted_indices = np.argsort(scores)
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]
        
        # Compute FAR and FRR for different thresholds
        n_genuine = len(genuine_scores)
        n_impostor = len(impostor_scores)
        
        far = []
        frr = []
        thresholds = []
        
        for threshold in np.linspace(scores.min(), scores.max(), 1000):
            # False Accept Rate: impostor scores >= threshold
            fa = np.sum((scores >= threshold) & (labels == 0))
            far.append(fa / n_impostor if n_impostor > 0 else 0)
            
            # False Reject Rate: genuine scores < threshold
            fr = np.sum((scores < threshold) & (labels == 1))
            frr.append(fr / n_genuine if n_genuine > 0 else 0)
            
            thresholds.append(threshold)
        
        far = np.array(far)
        frr = np.array(frr)
        thresholds = np.array(thresholds)
        
        # Find EER (where FAR = FRR)
        eer_idx = np.argmin(np.abs(far - frr))
        eer = (far[eer_idx] + frr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold
    
    def evaluate_verification(self, pairs_file: Path, max_pairs: int = 1000) -> dict:
        """
        Evaluate speaker verification performance
        
        Args:
            pairs_file: Path to verification pairs file
            max_pairs: Maximum number of pairs to evaluate
            
        Returns:
            Evaluation metrics
        """
        print(f"\n📊 Evaluating on {pairs_file.name}...")
        
        # Load verification pairs
        pairs = self.load_verification_pairs(pairs_file)
        
        # Limit pairs for faster evaluation
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]
        
        print(f"Loaded {len(pairs)} verification pairs")
        
        genuine_scores = []
        impostor_scores = []
        latencies = []
        
        for label, audio1_rel, audio2_rel in tqdm(pairs, desc="Evaluating"):
            try:
                # Construct full paths
                audio1_path = self.test_audio_path / audio1_rel
                audio2_path = self.test_audio_path / audio2_rel
                
                # Check if files exist
                if not audio1_path.exists() or not audio2_path.exists():
                    continue
                
                start_time = time.time()
                
                # Preprocess and extract embeddings
                segments1 = self.audio_preprocessor.preprocess(str(audio1_path))
                segments2 = self.audio_preprocessor.preprocess(str(audio2_path))
                
                if len(segments1) == 0 or len(segments2) == 0:
                    continue
                
                # Use first segment for evaluation
                emb1 = self.embedding_model.extract_embedding(segments1[0])
                emb2 = self.embedding_model.extract_embedding(segments2[0])
                
                # Normalize
                emb1 = self.embedding_model.normalize_embedding(emb1)
                emb2 = self.embedding_model.normalize_embedding(emb2)
                
                # Compute similarity
                similarity = self.embedding_model.compute_similarity(emb1, emb2, 'cosine')
                
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
                # Store score
                if label == 1:
                    genuine_scores.append(similarity)
                else:
                    impostor_scores.append(similarity)
                    
            except Exception as e:
                print(f"Error processing pair: {e}")
                continue
        
        # Convert to arrays
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        latencies = np.array(latencies)
        
        print(f"\n✓ Processed {len(genuine_scores)} genuine pairs")
        print(f"✓ Processed {len(impostor_scores)} impostor pairs")
        
        # Compute EER
        eer, eer_threshold = self.compute_eer(genuine_scores, impostor_scores)
        
        # Compute latency statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        results = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'num_genuine': len(genuine_scores),
            'num_impostor': len(impostor_scores),
            'genuine_mean': np.mean(genuine_scores),
            'genuine_std': np.std(genuine_scores),
            'impostor_mean': np.mean(impostor_scores),
            'impostor_std': np.std(impostor_scores)
        }
        
        return results

    def print_results(self, results: dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("📈 EVALUATION RESULTS")
        print("="*60)
        print(f"\n🎯 Equal Error Rate (EER): {results['eer']*100:.2f}%")
        print(f"   Target: < 3.00%")
        print(f"   Status: {'✓ PASS' if results['eer'] < 0.03 else '✗ FAIL'}")
        print(f"   EER Threshold: {results['eer_threshold']:.4f}")

        print(f"\n⚡ Latency Performance:")
        print(f"   Average: {results['avg_latency_ms']:.2f} ms")
        print(f"   95th percentile: {results['p95_latency_ms']:.2f} ms")
        print(f"   Maximum: {results['max_latency_ms']:.2f} ms")
        print(f"   Target: < 800 ms")
        print(f"   Status: {'✓ PASS' if results['avg_latency_ms'] < 800 else '✗ FAIL'}")

        print(f"\n📊 Score Statistics:")
        print(f"   Genuine pairs: {results['num_genuine']}")
        print(f"     Mean: {results['genuine_mean']:.4f} ± {results['genuine_std']:.4f}")
        print(f"   Impostor pairs: {results['num_impostor']}")
        print(f"     Mean: {results['impostor_mean']:.4f} ± {results['impostor_std']:.4f}")

        print("\n" + "="*60)

    def plot_score_distribution(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray,
                                save_path: str = "score_distribution.png"):
        """Plot score distribution"""
        plt.figure(figsize=(10, 6))

        plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='green')
        plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red')

        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Speaker Verification Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Score distribution plot saved to {save_path}")
        plt.close()

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("🚀 VOICEPRINT ANALYSIS SYSTEM EVALUATION")
        print("="*60)

        # Evaluate on verification test set
        results = self.evaluate_verification(self.veri_test, max_pairs=1000)

        # Print results
        self.print_results(results)

        # Save results
        import json
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")

        return results


def main():
    """Main training/evaluation function"""
    print("Initializing Voiceprint Analysis Trainer...")

    trainer = VoiceprintTrainer()

    # Run evaluation
    results = trainer.run_evaluation()

    # Check if requirements are met
    print("\n" + "="*60)
    print("✅ REQUIREMENTS CHECK")
    print("="*60)

    eer_pass = results['eer'] < 0.03
    latency_pass = results['avg_latency_ms'] < 800

    print(f"EER < 3%: {'✓ PASS' if eer_pass else '✗ FAIL'} ({results['eer']*100:.2f}%)")
    print(f"Latency < 800ms: {'✓ PASS' if latency_pass else '✗ FAIL'} ({results['avg_latency_ms']:.2f}ms)")

    if eer_pass and latency_pass:
        print("\n🎉 All requirements met! System is ready for deployment.")
    else:
        print("\n⚠️  Some requirements not met. Further optimization needed.")

    print("="*60)


if __name__ == "__main__":
    main()

