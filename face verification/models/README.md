# Model Checkpoint Directory

Place your trained ResNet50 triplet loss model checkpoint here:
- **Filename**: `best_model.pt`

## Model Requirements
- Architecture: ResNet50 with MLP embedding head (2048 -> 512 -> 128)
- Embedding dimension: 128
- Training: Triplet loss
- Input size: 112x112

## Training Your Model
If you need to train the model, use the triplet loss training script with:
- Anchor, Positive, Negative triplets
- Face dataset (LFW, VGGFace2, etc.)
- Learning rate: 0.0001
- Optimizer: Adam
- Margin: 0.2

The model will be automatically loaded when the backend starts.
