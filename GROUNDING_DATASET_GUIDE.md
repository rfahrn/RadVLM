# Grounding Dataset Creation for veRL GRPO Training

## Overview

This guide explains how to adapt the RadVLM codebase to create comprehensive grounding datasets for veRL GRPO training. The approach focuses on:

1. **Combining multiple grounding datasets** (MS-CXR, VinDr-CXR, PadChest-GR)
2. **IoU-based reward modeling** for grounding tasks
3. **veRL-compatible format** for GRPO training
4. **mAP evaluation** using RadVLM

## Why This Approach Works

### ✅ **Advantages:**
- **Comprehensive coverage**: Combines multiple medical imaging grounding datasets
- **IoU reward modeling**: Perfect for grounding tasks where spatial accuracy matters
- **veRL compatibility**: Follows the exact format required for GRPO training
- **Scalable evaluation**: Can evaluate using mAP metrics via RadVLM
- **Existing infrastructure**: Builds on proven RadVLM dataset handling

### ✅ **Technical Benefits:**
- **Normalized bounding boxes**: Consistent [x1, y1, x2, y2] format across datasets
- **Proper train/test splits**: Maintains dataset integrity for evaluation
- **Flexible reward modeling**: Easy to adjust IoU thresholds or add new metrics
- **Multi-dataset support**: Handles different annotation formats seamlessly

## Dataset Structure

The created datasets follow veRL's required format:

```python
{
    "data_source": "MS-CXR",
    "prompt": [
        {"role": "user", "content": "Please provide bounding box coordinates for: enlarged cardiac silhouette"}
    ],
    "ability": "phrase_grounding",
    "reward_model": {
        "style": "iou",
        "ground_truth": [[0.2, 0.3, 0.8, 0.7]],  # normalized coordinates
        "label": "enlarged cardiac silhouette"
    },
    "extra_info": {
        "image_path": "/path/to/image.jpg",
        "dataset_source": "MS-CXR",
        "split": "train",
        "sample_id": "MS-CXR_train_0"
    }
}
```

## Usage Instructions

### 1. **Create Grounding Dataset**

```bash
# Basic usage
python radvlm/data/create_comprehensive_grounding_dataset.py \
    --data_dir /path/to/datasets \
    --output_dir ./grounding_datasets

# With custom parameters
python radvlm/data/create_comprehensive_grounding_dataset.py \
    --data_dir /path/to/datasets \
    --output_dir ./grounding_datasets \
    --batch_size 64 \
    --num_workers 8 \
    --max_samples_per_dataset 10000
```

**Output:**
- `train.parquet`: Training dataset for GRPO
- `test.parquet`: Test dataset for evaluation

### 2. **Train with veRL GRPO**

Use the generated `train.parquet` file with veRL:

```python
# Example veRL training configuration
import datasets
from verl.trainer import GRPOTrainer

# Load the dataset
train_dataset = datasets.load_dataset('parquet', data_files='./grounding_datasets/train.parquet')

# Configure GRPO trainer
trainer = GRPOTrainer(
    model_name="your_model",
    dataset=train_dataset,
    reward_model_config={
        "style": "iou",
        "threshold": 0.5
    }
)

# Train
trainer.train()
```

### 3. **Evaluate Performance**

```bash
# Evaluate grounding performance
python radvlm/evaluation/evaluate_grounding_map.py \
    --predictions model_predictions.json \
    --ground_truth ./grounding_datasets/test.parquet \
    --output_dir ./evaluation_results
```

**Evaluation Metrics:**
- **mAP@0.3, mAP@0.5, mAP@0.7**: Mean Average Precision at different IoU thresholds
- **Mean IoU**: Average IoU across all predictions
- **Detection Rate**: Percentage of samples with valid predictions
- **Per-dataset metrics**: Performance breakdown by dataset source

## Dataset Sources

| Dataset | Task | Samples | Description |
|---------|------|---------|-------------|
| **MS-CXR** | Phrase grounding | ~1,000 | Chest X-ray phrase-level grounding |
| **VinDr-CXR** | Abnormality grounding | ~16,000 | Abnormality detection and localization |
| **PadChest-GR** | Phrase grounding | ~4,500 | Spanish chest X-ray grounding |

## Key Features

### **IoU-Based Reward Modeling**
- Calculates intersection over union between predicted and ground truth boxes
- Supports multiple IoU thresholds for different precision requirements
- Handles multiple bounding boxes per sample

### **Flexible Instruction Templates**
- Default: "Please provide bounding box coordinates for: {description}"
- Customizable for different prompting strategies
- Supports conversation-style interactions

### **Robust Evaluation**
- Handles various response formats (JSON, lists, plain text)
- Comprehensive metrics including mAP, IoU distribution, per-dataset performance
- Detailed error analysis and debugging support

## Advanced Configuration

### **Custom Reward Functions**
You can extend the reward modeling for more sophisticated metrics:

```python
def custom_reward_function(pred_boxes, gt_boxes, label):
    """Custom reward function for grounding tasks"""
    # Calculate IoU
    iou = calculate_best_iou(pred_boxes, gt_boxes)
    
    # Add penalty for multiple predictions when single expected
    if len(pred_boxes) > 1 and len(gt_boxes) == 1:
        iou *= 0.9
    
    # Bonus for exact matches
    if iou > 0.9:
        iou = min(1.0, iou + 0.05)
    
    return iou
```

### **Multi-Scale Evaluation**
For different image scales, you can normalize coordinates:

```python
def normalize_coordinates(boxes, image_width, image_height):
    """Normalize bounding box coordinates to [0, 1] range"""
    normalized = []
    for box in boxes:
        x1, y1, x2, y2 = box
        normalized.append([
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        ])
    return normalized
```

## Integration with RadVLM

After GRPO training, evaluate using RadVLM:

```python
# Load trained model
model = load_trained_model("path/to/grpo_model")

# Generate predictions on test set
predictions = []
for sample in test_dataset:
    response = model.generate(sample['prompt'])
    predictions.append({
        'sample_id': sample['extra_info']['sample_id'],
        'response': response
    })

# Evaluate
metrics = evaluate_grounding_performance(predictions, test_ground_truth)
```

## Troubleshooting

### **Common Issues:**

1. **Empty bounding boxes**: Check that your datasets have valid box annotations
2. **Coordinate format**: Ensure boxes are in [x1, y1, x2, y2] normalized format
3. **Memory issues**: Reduce batch_size or max_samples_per_dataset
4. **Missing datasets**: Verify DATA_DIR contains all required dataset directories

### **Performance Tips:**

- Use `--max_samples_per_dataset` for quick testing
- Increase `--num_workers` for faster data loading
- Balance datasets by adjusting sampling rates
- Use GPU for faster inference during evaluation

## Conclusion

This approach provides a robust foundation for creating comprehensive grounding datasets for veRL GRPO training. The combination of multiple medical imaging datasets, IoU-based reward modeling, and comprehensive evaluation metrics makes it well-suited for developing high-quality grounding models in the medical domain.

The integration with RadVLM ensures that you can leverage the existing infrastructure while adapting it for modern RLHF training approaches.