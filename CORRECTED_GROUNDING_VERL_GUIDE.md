# ⚠️ CORRECTED: RadVLM Grounding Dataset for veRL GRPO Training

## 🚨 **Critical Issues Found in Existing Code**

After thorough analysis, the existing `create_verl_dataset_llava.py` has **serious issues** for GRPO training:

1. **❌ INCLUDES ANSWER IN PROMPT**: For GRPO, we want only the question - the model should generate the answer and be rewarded based on IoU
2. **❌ WRONG FORMAT**: Current format includes both user and assistant turns
3. **❌ MISSING REQUIRED FIELDS**: No 'split' and 'index' in extra_info as required by veRL
4. **❌ HARDCODED DATA_SOURCE**: Always "RadVLM" instead of dataset-specific names

## ✅ **Corrected Solution**

### **1. Create Corrected Dataset**

```bash
# Use the corrected script
python radvlm/data/create_grounding_verl_corrected.py \
    --data_dir /path/to/datasets \
    --output_dir ./grounding_datasets_corrected \
    --splits train test
```

### **2. Format Comparison**

**❌ What the existing code produces (WRONG for GRPO):**
```python
{
    "data_source": "RadVLM",
    "prompt": [
        {"role": "user", "content": "Please locate: enlarged cardiac silhouette"},
        {"role": "assistant", "content": "Located at [0.2, 0.3, 0.8, 0.7]"}  # ❌ WRONG!
    ],
    "extra_info": {
        "image_path": "/path/to/image.jpg",
        "id_source": "ms-cxr_0"  # ❌ Missing split/index
    }
}
```

**✅ What the corrected code produces (CORRECT for GRPO):**
```python
{
    "data_source": "MS-CXR",  # ✅ Dataset-specific
    "prompt": [
        {"role": "user", "content": "Please locate: enlarged cardiac silhouette"}
        # ✅ NO ANSWER - model generates it and gets rewarded based on IoU
    ],
    "ability": "phrase_grounding",
    "reward_model": {
        "style": "iou",
        "ground_truth": [[0.2, 0.3, 0.8, 0.7]],
        "label": "enlarged cardiac silhouette"
    },
    "extra_info": {
        "split": "train",         # ✅ Required by veRL
        "index": 0,              # ✅ Required by veRL
        "image_path": "/path/to/image.jpg",
        "dataset_source": "MS-CXR",
        "sample_id": "MS-CXR_train_0"
    }
}
```

### **3. Train with veRL GRPO**

```python
# Now the format is correct for GRPO training
import datasets
from verl.trainer import GRPOTrainer

# Load the corrected dataset
train_dataset = datasets.load_dataset('parquet', data_files='./grounding_datasets_corrected/train.parquet')

# Configure GRPO trainer
trainer = GRPOTrainer(
    model_name="your_model",
    dataset=train_dataset,
    reward_model_config={
        "style": "iou",
        "threshold": 0.5
    }
)

# Train with proper reinforcement learning setup
trainer.train()
```

### **4. Evaluate with Existing RadVLM Infrastructure**

The evaluation infrastructure is still valid:

```bash
# RadVLM evaluation still works correctly
accelerate launch --num_processes=4 radvlm.evaluation.evaluate_instructions \
    --task phrase_grounding \
    --model_name your_trained_model_path
```

## 🔍 **Key Differences in Corrected Version**

### **Prompt Generation**
- **Original**: Includes both question and answer
- **Corrected**: Only question - model generates answer and gets IoU reward

### **Data Source**
- **Original**: Always "RadVLM"
- **Corrected**: Dataset-specific ("MS-CXR", "VinDr-CXR", "PadChest-GR")

### **Extra Info**
- **Original**: Missing 'split' and 'index'
- **Corrected**: Includes all required veRL fields

### **Question Templates**
- **Original**: Uses RadVLM's instruction generation
- **Corrected**: GRPO-optimized templates focused on bounding box prediction

## 🎯 **Why This Matters for GRPO**

1. **Reinforcement Learning**: Model must generate answers to learn from rewards
2. **IoU Rewards**: Only meaningful if model generates the coordinates
3. **Policy Optimization**: GRPO requires proper action space (generated text)
4. **Training Efficiency**: Correct format enables proper gradient flow

## 📊 **Expected Dataset Statistics**

| Dataset | Train Samples | Test Samples | Task Type |
|---------|--------------|-------------|-----------|
| MS-CXR | ~971 | ~189 | Phrase grounding |
| VinDr-CXR | ~16,089 | ~2,108 | Abnormality grounding |
| VinDr-CXR-Single | ~15,000 | ~2,000 | Single-label grounding |
| PadChest-GR | ~4,478 | ~validation | Phrase grounding |

## 🚀 **Complete Workflow**

1. **Create corrected datasets**:
   ```bash
   python radvlm/data/create_grounding_verl_corrected.py --data_dir /path/to/datasets
   ```

2. **Train with veRL GRPO**:
   ```python
   # Use train.parquet with correct format
   ```

3. **Evaluate with RadVLM**:
   ```bash
   accelerate launch --num_processes=4 radvlm.evaluation.evaluate_instructions --task phrase_grounding --model_name your_model
   ```

## ⚠️ **Important Notes**

- **DO NOT use the original `create_verl_dataset_llava.py`** - it's incompatible with GRPO
- **Use the corrected version** for proper veRL training
- **Evaluation infrastructure is still valid** - RadVLM's mAP calculation works correctly
- **The corrected format follows veRL specifications** exactly

## 🔧 **Testing the Corrected Format**

You can verify the format is correct by checking:
1. Prompt contains only user's question
2. No assistant response in prompt
3. extra_info includes 'split' and 'index'
4. data_source is dataset-specific
5. reward_model has proper IoU setup

This corrected approach ensures your GRPO training will work properly with veRL and the evaluation will work correctly with RadVLM!