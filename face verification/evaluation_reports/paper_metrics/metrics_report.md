# Face Verification Metrics Report

## Evaluated Data
- Split folder: C:\Users\hivin\Desktop\Cyber_Project_1121\face verification\Dataset\dataset\test
- Number of identities: 54
- Number of pairs: 864
- Genuine pairs (label=1): 432
- Impostor pairs (label=0): 432

## Metric Definitions
- Accuracy: Overall fraction of correct decisions.
- Precision: Among predicted matches, how many are true matches.
- Recall (TPR): Among true matches, how many were correctly accepted.
- Specificity (TNR): Among true non-matches, how many were correctly rejected.
- Confusion Matrix: Counts of TN, FP, FN, TP.
- FAR (False Acceptance Rate): FP / (FP + TN). Lower is better for security.
- FRR (False Rejection Rate): FN / (FN + TP). Lower is better for usability.
- EER (Equal Error Rate): Error rate where FAR and FRR are equal (or closest). Lower is better.

## Results
- Threshold: 0.809600
- Accuracy: 0.500000
- Precision: 0.500000
- Recall: 0.949074
- Specificity: 0.050926
- FAR: 0.949074
- FRR: 0.050926
- EER: 0.500000
- EER threshold: 0.949794

## Confusion Matrix
- TN: 22
- FP: 410
- FN: 22
- TP: 410