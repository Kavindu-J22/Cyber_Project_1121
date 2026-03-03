# Model Checkpoint Directory

Place your trained ResNet50 triplet loss model checkpoint here:
- **Filename**: `best_resnet50_triplet.pth`

## Model Requirements
- Architecture: ResNet50 with custom embedding layer
- Embedding dimension: 128
- Training: Triplet loss
- Input size: 224x224

## Training Your Model
If you need to train the model, use the triplet loss training script with:
- Anchor, Positive, Negative triplets
- Face dataset (LFW, VGGFace2, etc.)
- Learning rate: 0.001
- Optimizer: Adam
- Margin: 0.2-0.5

The model will be automatically loaded when the backend starts.
