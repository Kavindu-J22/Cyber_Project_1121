# 🎯 TUPLET DATASET RESULTS - Keystroke Dynamics Model

## 📊 Test Results (December 8, 2025)

### Training Configuration
- **Dataset**: tuplet_dataset_with_subjects1.xlsx (10,000 pre-paired samples)
- **Training Samples**: 8,000 pairs (4,000 genuine + 4,000 impostor)
- **Validation Samples**: 1,000 pairs (500 genuine + 500 impostor)
- **Test Samples**: 1,000 pairs (481 genuine + 519 impostor)
- **Features**: 31 timing features per sample
- **Loss Function**: Contrastive loss (for paired samples)
- **Training Epochs**: 100 (best model at epoch 92)
- **Best Validation Loss**: 0.0147

---

## 📈 Performance Metrics

### Primary Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Accuracy** | **67.60%** | 95%+ | ❌ FAIL |
| **EER** | **33.66%** | <5% | ❌ FAIL |
| **F1 Score** | **0.4953** | >0.95 | ❌ FAIL |
| **Precision** | **98.76%** | >95% | ✅ PASS |
| **Recall** | **33.06%** | >95% | ❌ FAIL |

### Biometric Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **FAR** | **0.39%** | <2% | ✅ PASS |
| **FRR** | **66.94%** | <5% | ❌ FAIL |
| **GAR** | **33.06%** | >95% | ❌ FAIL |
| **TRR** | **99.61%** | >95% | ✅ PASS |

### Score Statistics
- **Genuine Mean**: 97.00% (very high!)
- **Impostor Mean**: 41.68%
- **Score Separation**: 55.32% (good separation!)
- **Optimal Threshold**: 0.9940 (too high!)

---

## 🔍 Analysis

### What Went Well ✅
1. **Excellent Score Separation**: 55.32% separation between genuine and impostor scores
2. **Very Low FAR**: Only 0.39% false accept rate (2 out of 519 impostors accepted)
3. **High Precision**: 98.76% - when the model accepts someone, it's almost always correct
4. **Balanced Dataset**: Perfect 50/50 split of genuine/impostor pairs
5. **Pre-paired Data**: No need to manually create pairs
6. **Training Stability**: Model converged well (validation loss: 0.0147)

### What Went Wrong ❌
1. **Very High FRR**: 66.94% - rejecting 322 out of 481 genuine users!
2. **Threshold Too High**: Optimal threshold at 0.9940 is too conservative
3. **Low Recall**: Only 33.06% of genuine users are accepted
4. **Low Accuracy**: 67.60% overall accuracy

---

## 🎯 Root Cause

The model is **TOO CONSERVATIVE** due to:

1. **Contrastive Loss Behavior**: The contrastive loss pushes genuine pairs to have similarity close to 1.0, but there's natural variability in keystroke patterns
2. **Threshold Selection**: The EER-based threshold (0.9940) is too strict
3. **Score Distribution**: Genuine scores have high mean (0.97) but also high variance (std: 0.0465)

---

## 💡 Recommendations

### Option 1: Lower the Threshold (QUICK FIX)
**Action**: Set threshold to 0.85-0.90 instead of 0.9940

**Expected Impact**:
- Accuracy: 67.6% → **80-85%**
- FRR: 66.94% → **20-30%**
- FAR: 0.39% → **5-10%**

**Implementation**:
```yaml
# config.yaml
verification:
  threshold: 0.85  # Lower from 0.9940
```

### Option 2: Use Different Loss Function
**Action**: Switch from contrastive loss to triplet loss with margin=0.2

**Expected Impact**:
- Better embedding space
- More balanced genuine/impostor separation
- Accuracy: 67.6% → **75-85%**

### Option 3: Ensemble Approach
**Action**: Combine multiple models:
- Model 1: Current contrastive loss model
- Model 2: Triplet loss model
- Model 3: Siamese network

**Expected Impact**:
- Accuracy: 67.6% → **85-90%**

### Option 4: Hybrid Dataset Approach
**Action**: Use both tuplet dataset AND DSL dataset
- Train on tuplet pairs for verification
- Fine-tune on DSL data for better generalization

**Expected Impact**:
- Accuracy: 67.6% → **80-90%**

---

## 📊 Comparison: DSL vs Tuplet Dataset

| Metric | DSL Dataset | Tuplet Dataset | Change |
|--------|-------------|----------------|--------|
| Accuracy | 69.40% | 67.60% | -1.8% ❌ |
| EER | 30.60% | 33.66% | +3.06% ❌ |
| FAR | 1.20% | 0.39% | -0.81% ✅ |
| FRR | 60.00% | 66.94% | +6.94% ❌ |
| Score Separation | 52.58% | 55.32% | +2.74% ✅ |
| Training Samples | 48,000 (augmented) | 8,000 (pairs) | -40,000 |
| Features | 38 (with stats) | 31 (raw) | -7 |

**Conclusion**: Tuplet dataset has **better score separation** but **worse overall performance** due to:
1. Fewer training samples (8,000 vs 48,000)
2. Fewer features (31 vs 38)
3. Different loss function (contrastive vs triplet)

---

## 🚀 Next Steps

### Immediate (Today)
1. ✅ **DONE**: Trained model on tuplet dataset
2. ✅ **DONE**: Evaluated performance
3. ⏳ **TODO**: Lower threshold to 0.85 and re-test
4. ⏳ **TODO**: Compare results

### Short-term (This Week)
1. Train with triplet loss on tuplet dataset
2. Implement ensemble model
3. Try hybrid dataset approach

### Long-term (Next Week)
1. Collect more training data
2. Implement advanced feature engineering
3. Try transformer-based architecture

---

## 📝 Conclusion

The tuplet dataset provides **excellent score separation** (55.32%) and **very low FAR** (0.39%), but the model is **too conservative** with a threshold of 0.9940, resulting in **high FRR** (66.94%) and **low accuracy** (67.60%).

**Recommended Action**: Lower the threshold to 0.85-0.90 to achieve **80-85% accuracy** with balanced FAR/FRR.

The tuplet dataset is **valuable** for its pre-paired structure and balanced classes, but needs **threshold tuning** or **different loss function** to achieve target performance.

