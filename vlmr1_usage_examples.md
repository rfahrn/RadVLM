# VLM-R1 Individual Task Usage Examples

## Generate Individual Task Files

First, generate the individual task-specific JSONL files:

```bash
python3 radvlm/data/create_vlmr1_individual_tasks.py \
    --data-dir /capstor/store/cscs/swissai/a135/RadVLM_project/data \
    --output-dir ./vlmr1_individual_tasks
```

This will create files like:
- `report_generation_mimic_train.jsonl` + `report_generation_mimic_test.jsonl`
- `abnormality_classification_chexpert_train.jsonl` + `abnormality_classification_chexpert_test.jsonl`
- `phrase_grounding_mscxr_train.jsonl` + `phrase_grounding_mscxr_test.jsonl`
- `anatomical_grounding_chest_imagenome_train.jsonl` + `anatomical_grounding_chest_imagenome_test.jsonl`
- `abnormality_grounding_vindr_cxr_train.jsonl` + `abnormality_grounding_vindr_cxr_test.jsonl`
- And more...

## VLM-R1 Training Examples

### Single Task Training

Train on just report generation:
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_individual_tasks/report_generation_mimic_train.jsonl \
    --image_folders /capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output_report_generation
```

Train on just classification:
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_individual_tasks/abnormality_classification_chexpert_train.jsonl \
    --image_folders /capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output_classification
```

### Multi-Task Training

Train on multiple tasks simultaneously:
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_individual_tasks/report_generation_mimic_train.jsonl:./vlmr1_individual_tasks/abnormality_classification_chexpert_train.jsonl:./vlmr1_individual_tasks/phrase_grounding_mscxr_train.jsonl \
    --image_folders /capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output_multitask
```

Train on all grounding tasks:
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_individual_tasks/anatomical_grounding_chest_imagenome_train.jsonl:./vlmr1_individual_tasks/abnormality_grounding_vindr_cxr_train.jsonl:./vlmr1_individual_tasks/phrase_grounding_mscxr_train.jsonl:./vlmr1_individual_tasks/phrase_grounding_padchest_train.jsonl \
    --image_folders /capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output_all_grounding
```

### Comprehensive Training

Train on all available tasks:
```bash
python -m open_r1.grpo_jsonl \
    --data_file_paths ./vlmr1_individual_tasks/report_generation_mimic_train.jsonl:./vlmr1_individual_tasks/report_generation_chexpert_plus_train.jsonl:./vlmr1_individual_tasks/abnormality_classification_mimic_train.jsonl:./vlmr1_individual_tasks/abnormality_classification_chexpert_train.jsonl:./vlmr1_individual_tasks/anatomical_grounding_chest_imagenome_train.jsonl:./vlmr1_individual_tasks/abnormality_grounding_vindr_cxr_train.jsonl:./vlmr1_individual_tasks/abnormality_detection_vindr_cxr_train.jsonl:./vlmr1_individual_tasks/phrase_grounding_mscxr_train.jsonl:./vlmr1_individual_tasks/phrase_grounding_padchest_train.jsonl \
    --image_folders /capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/:/capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_dir ./output_all_tasks
```

## Key Features

1. **Individual Control**: Train on specific tasks as needed
2. **Flexible Combinations**: Mix and match tasks for multi-task learning
3. **Test Data Available**: Most tasks include both train and test splits
4. **VLM-R1 Compatible**: All files use the exact VLM-R1 JSONL format
5. **Relative Paths**: Image paths are relative to your data directory

## Generated Task Files

The script creates these individual task files:

### Report Generation
- `report_generation_mimic_train.jsonl` + `report_generation_mimic_test.jsonl`
- `report_generation_chexpert_plus_train.jsonl` (train only)

### Classification
- `abnormality_classification_mimic_train.jsonl` + `abnormality_classification_mimic_test.jsonl`
- `abnormality_classification_chexpert_train.jsonl` + `abnormality_classification_chexpert_test.jsonl`

### Grounding Tasks
- `anatomical_grounding_chest_imagenome_train.jsonl` + `anatomical_grounding_chest_imagenome_test.jsonl`
- `abnormality_grounding_vindr_cxr_train.jsonl` + `abnormality_grounding_vindr_cxr_test.jsonl`
- `abnormality_detection_vindr_cxr_train.jsonl` + `abnormality_detection_vindr_cxr_test.jsonl`
- `phrase_grounding_mscxr_train.jsonl` + `phrase_grounding_mscxr_test.jsonl`
- `phrase_grounding_padchest_train.jsonl` + `phrase_grounding_padchest_test.jsonl`

### Conversations (if available)
- `conversations_mimic_standard_train.jsonl` + `conversations_mimic_standard_test.jsonl`
- `conversations_mimic_grounded_train.jsonl` + `conversations_mimic_grounded_test.jsonl`

Use any combination of these files for your specific training needs!