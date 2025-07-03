# RadVLM to VLM-R1 Dataset Conversion

This guide explains how to convert RadVLM datasets to VLM-R1 compatible JSONL format for use with Qwen 2.5-VL and VLM-R1 training.

## Prerequisites

1. **RadVLM Installation**: Ensure you have RadVLM properly installed and configured
2. **Python Dependencies**: Install required packages:
   ```bash
   pip install -r vlmr1_requirements.txt
   ```
3. **Dataset Setup**: Have your medical imaging datasets configured in RadVLM's expected directory structure

## Overview

The scripts in this repository convert RadVLM's multi-modal medical datasets into the JSONL format expected by VLM-R1. The conversion handles:

- **Single and multi-image datasets**
- **All RadVLM dataset types** (MIMIC-CXR, CheXpert, PadChest, VinDr-CXR, etc.)
- **Different conversation formats** (instructions, conversations, grounding tasks)
- **Proper image path handling** (relative paths as required by VLM-R1)

## VLM-R1 Format

The generated JSONL files follow the VLM-R1 specification:

### Single Image Format
```json
{
  "id": 1,
  "image": "MIMIC-CXR-JPG/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
  "conversations": [
    {"from": "human", "value": "<image>What number of purple metallic balls are there?"},
    {"from": "gpt", "value": "0"}
  ]
}
```

### Multi-Image Format
```json
{
  "id": 2,
  "image": ["path/to/image1.jpg", "path/to/image2.jpg"],
  "conversations": [
    {"from": "human", "value": "<image><image>What number of purple metallic balls in total within the two images?"},
    {"from": "gpt", "value": "3"}
  ]
}
```

## Scripts

### 1. `create_vlmr1_comprehensive.py`
Main script for generating training datasets from all RadVLM sources.

**Usage:**
```bash
# Generate all training datasets separately
python radvlm/data/create_vlmr1_comprehensive.py \
    --data-dir /path/to/your/datasets \
    --output-dir ./vlmr1_datasets \
    --batch-size 64 \
    --num-workers 8

# Generate a single combined training file
python radvlm/data/create_vlmr1_comprehensive.py \
    --data-dir /path/to/your/datasets \
    --output-dir ./vlmr1_datasets \
    --combine-all

# Generate specific datasets only
python radvlm/data/create_vlmr1_comprehensive.py \
    --data-dir /path/to/your/datasets \
    --output-dir ./vlmr1_datasets \
    --datasets mimic_reports chexpert vindr_cxr

# Test with limited samples
python radvlm/data/create_vlmr1_comprehensive.py \
    --data-dir /path/to/your/datasets \
    --output-dir ./vlmr1_datasets \
    --limit 100
```

### 2. `create_vlmr1_test.py`
Script for generating test/validation datasets.

**Usage:**
```bash
# Generate test datasets
python radvlm/data/create_vlmr1_test.py \
    --data-dir /path/to/your/datasets \
    --output-dir ./vlmr1_datasets \
    --combine-all
```

## Supported Datasets

The scripts support all RadVLM datasets:

| Dataset | Type | Train Script | Test Script |
|---------|------|--------------|-------------|
| MIMIC-CXR Reports | Report Generation | ✅ | ✅ |
| MIMIC-CXR Classification | Classification | ✅ | ✅ |
| CheXpert | Classification | ✅ | ✅ |
| CheXpert-Plus | Report Generation | ✅ | ❌ |
| Chest-ImaGenome | Region Grounding | ✅ | ❌ |
| VinDr-CXR | Detection | ✅ | ✅ |
| VinDr-CXR Single Label | Classification | ✅ | ✅ |
| MS-CXR | Phrase Grounding | ✅ | ❌ |
| PadChest Grounding | Phrase Grounding | ✅ | ✅ |
| MIMIC Conversations | Conversations | ✅ | ✅ |
| PadChest Conversations | Conversations | ✅ | ❌ |

## Parameters

### Common Parameters
- `--data-dir`: Root directory containing all datasets (defaults to RadVLM's DATA_DIR)
- `--output-dir`: Directory to save JSONL files (default: `./vlmr1_datasets`)
- `--batch-size`: Batch size for data loading (default: 64)
- `--num-workers`: Number of workers for data loading (default: 8)
- `--seed`: Random seed for reproducibility (default: 0)
- `--shuffle`: Shuffle datasets during processing
- `--combine-all`: Create single combined JSONL file instead of separate files
- `--datasets`: Process only specific datasets (list of dataset names)
- `--limit`: Limit number of samples per dataset (useful for testing)

## Output Files

### Individual Dataset Files
When `--combine-all` is not used, separate files are created:
- `mimic_reports_train.jsonl`
- `mimic_labels_train.jsonl`
- `chexpert_train.jsonl`
- `vindr_cxr_train.jsonl`
- etc.

### Combined Files
When `--combine-all` is used:
- `all_train.jsonl` (from training script)
- `all_test.jsonl` (from test script)

## Using with VLM-R1

Once you have generated the JSONL files, use them with VLM-R1 training:

```bash
# Single dataset
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_datasets/all_train.jsonl \
    --image_folders /path/to/your/datasets/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output

# Multiple datasets
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_datasets/all_train.jsonl:./vlmr1_datasets/all_test.jsonl \
    --image_folders /path/to/datasets1/:/path/to/datasets2/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output
```

## Important Notes

### Image Paths
- All image paths in the JSONL files are **relative** to the `--image_folders` parameter
- The scripts automatically convert absolute paths from RadVLM to relative paths
- Ensure your `--image_folders` parameter points to the correct base directory

### Data Directory Structure
Your data directory should follow the RadVLM structure:
```
data_dir/
├── MIMIC-CXR-JPG/
├── CheXpert/
├── PadChest/
├── VinDr-CXR/
├── MS-CXR/
└── CHEST_IMA/
```

### Memory Considerations
- Use appropriate `--batch-size` and `--num-workers` based on your system
- Large datasets may require significant memory; consider using `--limit` for testing
- The `--shuffle` option can help with memory distribution during processing

## Troubleshooting

### Missing Datasets
If certain datasets are not available, the scripts will skip them gracefully and continue processing available datasets.

### Path Issues
Ensure all dataset paths are correctly configured in your RadVLM installation and that the `DATA_DIR` environment variable is set.

### Memory Issues
Reduce `--batch-size` and `--num-workers` if you encounter memory problems.

## Example Workflow

1. **Test with small sample:**
```bash
python radvlm/data/create_vlmr1_comprehensive.py \
    --output-dir ./test_output \
    --limit 10 \
    --combine-all
```

2. **Generate full training dataset:**
```bash
python radvlm/data/create_vlmr1_comprehensive.py \
    --output-dir ./vlmr1_datasets \
    --combine-all
```

3. **Generate test dataset:**
```bash
python radvlm/data/create_vlmr1_test.py \
    --output-dir ./vlmr1_datasets \
    --combine-all
```

4. **Train with VLM-R1:**
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_datasets/all_train.jsonl \
    --image_folders /path/to/datasets/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct"
```

This conversion pipeline enables you to leverage all of RadVLM's medical imaging datasets with VLM-R1 and Qwen 2.5-VL for advanced vision-language model training.