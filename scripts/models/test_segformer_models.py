# %%
# Setup
import torch
import torch.nn as nn
from transformers import (
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)

MODEL_DIR = "/scratch/models/nvidia/mit-b2"

# %%
# Setup classification model
model = SegformerForImageClassification.from_pretrained(
    MODEL_DIR, num_labels=1, ignore_mismatched_sizes=True)

# %%
# Verify classification model configuration
print(model.config)

# %%
print(model)

# %%
# Test model with a random input
batch = torch.randn(2, 3, 224, 224)
output = model(pixel_values=batch)
print(type(output))
print(output.logits.shape)

# %%
# Setup segmentation model
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR,
    num_labels=5,
)

# %%
# Verify segmentation model configuration
print(model.config)

# %%
# Verify segmentation model architecture
print(model)

# %%
# Test model with a random input
batch = torch.randn(2, 3, 224, 224)
output = model(pixel_values=batch)
print(type(output))
print(output.logits.shape)

upsampled_logits = nn.functional.interpolate(
    output.logits,
    size=(224, 224),  # (height, width)
    mode='bilinear',
    align_corners=False
)
print(upsampled_logits.shape)


# %%
