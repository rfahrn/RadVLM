
"""
Flexible VLM-R1 dataset generator allowing specific task and dataset selection.
Based on RadVLM paper dataset table - allows precise control over what gets generated.

Usage examples:
  --task phrase_grounding --dataset MS-CXR
  --task phrase_grounding --dataset both
  --task abnormality_grounding
  --dataset PadChest-GR
  --task report_generation --dataset MIMIC-CXR
"""

import torch
import numpy as np
import random
from radvlm.data.datasets import MIMIC_Dataset_MM, CheXpert_Dataset_MM, Chest_ImaGenome_Dataset, MS_CXR, CheXpertPlus_Dataset, PadChest_grounding, PadChest_grounding_per_image, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset
from radvlm.data.utils import *
import json
import os
import argparse
from torch.utils.data import ConcatDataset, DataLoader
from radvlm import DATA_DIR


# RadVLM paper task-dataset mapping
TASK_DATASET_MAPPING = {
    "report_generation": {
        "MIMIC-CXR": {"samples": 230980, "multiplier": 1, "eval": 3314},
        "CheXpert-Plus": {"samples": 186463, "multiplier": 1, "eval": None}
    },
    "abnormality_classification": {
        "MIMIC-CXR": {"samples": 237912, "multiplier": 1, "eval": 518},
        "CheXpert": {"samples": 191027, "multiplier": 1, "eval": None}
    },
    "anatomical_grounding": {
        "Chest-ImaGenome": {"samples": 80000, "multiplier": 1, "eval": 2000}
    },
    "abnormality_grounding": {
        "VinDr-CXR": {"samples": 16089, "multiplier": 3, "eval": 2108}
    },
    "abnormality_detection": {
        "VinDr-CXR": {"samples": 15000, "multiplier": 2, "eval": None}
    },
    "phrase_grounding": {
        "MS-CXR": {"samples": 971, "multiplier": 3, "eval": 189},
        "PadChest-GR": {"samples": 4478, "multiplier": 2, "eval": None}
    },
    "conversation": {
        "MIMIC-CXR": {"samples": 86155, "multiplier": 1, "eval": 500}
    },
    "conversation_grounded": {
        "MS-CXR": {"samples": 862, "multiplier": 4, "eval": 155},
        "PadChest-GR": {"samples": 2225, "multiplier": 4, "eval": None}
    }
}

# Reverse mapping: dataset -> tasks
DATASET_TASK_MAPPING = {}
for task, datasets in TASK_DATASET_MAPPING.items():
    for dataset in datasets:
        if dataset not in DATASET_TASK_MAPPING:
            DATASET_TASK_MAPPING[dataset] = []
        DATASET_TASK_MAPPING[dataset].append(task)


def create_json_cell_vlmr1(sample, id_prefix, sample_idx, dataset, base_dir):
    """Create VLM-R1 compatible JSON cell from RadVLM sample."""
    sample_instr = sample["conversation"] if "conversation" in sample else sample["instr"]
    
    # Make image path relative to base_dir
    img_path = sample["img_path"]
    if isinstance(img_path, list):
        image_paths = [os.path.relpath(path, base_dir) for path in img_path]
        image_count = len(image_paths)
    else:
        image_paths = os.path.relpath(img_path, base_dir)
        image_count = 1
    
    json_cell = {
        "id": f"{id_prefix}_{sample_idx}",
        "image": image_paths,
        "conversations": []
    }

    if isinstance(sample_instr, dict):
        sample_instr = [sample_instr]
    
    for j, instr in enumerate(sample_instr):
        if "from" in instr and "value" in instr:
            conv_cell = instr.copy()
            if j == 0:
                image_tokens = "<image>" * image_count
                conv_cell["value"] = f"{image_tokens}{conv_cell['value']}"
            json_cell["conversations"].append(conv_cell)
        elif "question" in instr and "answer" in instr:
            conv_cell_human = {
                "from": "human",
                "value": instr["question"]
            }
            if j == 0:
                image_tokens = "<image>" * image_count
                conv_cell_human["value"] = f"{image_tokens}{conv_cell_human['value']}"
            
            conv_cell_ai = {
                "from": "gpt",
                "value": instr["answer"]
            }
            json_cell["conversations"].append(conv_cell_human)
            json_cell["conversations"].append(conv_cell_ai)

    return json_cell


def create_dataset(task, dataset_name, data_dir, split="train"):
    """Create specific dataset based on task and dataset name."""
    
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    
    # Report Generation
    if task == "report_generation":
        if dataset_name == "MIMIC-CXR":
            return MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split=split, 
                flag_img=False, flag_lab=False, 
                only_frontal=True, 
                filtered_reports_dir=filtered_reports_dir, 
                seed=0
            )
        elif dataset_name == "CheXpert-Plus":
            datasetpath_chex = os.path.join(data_dir, 'CheXpert')
            filtered_reports_dir_chex = os.path.join(datasetpath_chex, 'filtered_reports')
            return CheXpertPlus_Dataset(
                datasetpath=datasetpath_chex, 
                split=split, 
                flag_img=False, 
                filtered_reports_dir=filtered_reports_dir_chex
            )
    
    # Abnormality Classification
    elif task == "abnormality_classification":
        if dataset_name == "MIMIC-CXR":
            return MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split=split, 
                flag_img=False, flag_lab=True, 
                only_frontal=True, 
                filtered_reports_dir=None, 
                classif=True,
                seed=0
            )
        elif dataset_name == "CheXpert":
            dataset_path = os.path.join(data_dir, "CheXpert")
            return CheXpert_Dataset_MM(datasetpath=dataset_path, split=split, flag_img=False)
    
    # Anatomical Grounding
    elif task == "anatomical_grounding":
        if dataset_name == "Chest-ImaGenome":
            datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
            return Chest_ImaGenome_Dataset(
                datasetpath=datasetpath_mimic,
                datasetpath_chestima=datasetpath_chestima,
                split=split, 
                flag_img=False, 
                flag_instr=True, 
                flag_txt=False, 
                flag_lab=False,
                pick_one_region=True,
            )
    
    # Abnormality Grounding
    elif task == "abnormality_grounding":
        if dataset_name == "VinDr-CXR":
            dataset_path = os.path.join(data_dir, "VinDr-CXR")
            return VinDr_CXR_Dataset(datasetpath=dataset_path, split=split, flag_img=False)
    
    # Abnormality Detection
    elif task == "abnormality_detection":
        if dataset_name == "VinDr-CXR":
            dataset_path = os.path.join(data_dir, "VinDr-CXR")
            return VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split=split, flag_img=False)
    
    # Phrase Grounding
    elif task == "phrase_grounding":
        if dataset_name == "MS-CXR":
            return MS_CXR(
                datasetpath=datasetpath_mimic,
                split=split, flag_img=False, 
                flag_lab=True, only_frontal=True, 
                flag_instr=True, 
                sentencesBBoxpath=sentencesBBoxpath,
                seed=0
            )
        elif dataset_name == "PadChest-GR":
            datasetpath_pad = os.path.join(data_dir, 'PadChest')
            return PadChest_grounding(
                datasetpath=datasetpath_pad,
                split=split, 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
    
    # Conversation
    elif task == "conversation":
        if dataset_name == "MIMIC-CXR":
            conversation_dir = os.path.join(datasetpath_mimic, f'conversations/{split}/standard')
            if os.path.exists(conversation_dir):
                return MIMIC_Dataset_MM(
                    datasetpath=datasetpath_mimic,
                    split=split, 
                    flag_img=False, 
                    flag_instr=False, 
                    flag_txt=False, 
                    flag_lab=False, 
                    filtered_reports_dir=filtered_reports_dir,
                    conversation_dir=conversation_dir
                )
            return None
    
    # Conversation Grounded
    elif task == "conversation_grounded":
        if dataset_name == "MS-CXR":
            conversation_dir = os.path.join(datasetpath_mimic, f'conversations/{split}/grounding')
            if os.path.exists(conversation_dir):
                return MIMIC_Dataset_MM(
                    datasetpath=datasetpath_mimic,
                    split=split, flag_img=False, 
                    flag_lab=False, only_frontal=True, 
                    flag_instr=False, 
                    filtered_reports_dir=filtered_reports_dir,
                    sentencesBBoxpath=sentencesBBoxpath,
                    conversation_dir=conversation_dir,
                    classif=False,
                    seed=0
                )
            return None
        elif dataset_name == "PadChest-GR":
            datasetpath_pad = os.path.join(data_dir, 'PadChest')
            conversation_dir = os.path.join(datasetpath_pad, f'conversations/{split}/grounding')
            if os.path.exists(conversation_dir):
                return PadChest_grounding_per_image(
                    datasetpath=datasetpath_pad,
                    split=split,
                    flag_instr=False,
                    flag_img=False,
                    conversation_dir=conversation_dir
                )
            return None
    
    return None


def generate_vlmr1_samples(dataset, id_prefix, base_dir, num_samples=None, batch_size=64, num_workers=8, seed=0):
    """Generate VLM-R1 samples from dataset."""
    np.random.seed(seed)
    random.seed(seed)
    
    if num_samples is None:
        num_samples = len(dataset)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    samples = []
    sample_count = 0
    
    for batch in data_loader:
        for sample in batch:
            if sample_count >= num_samples:
                break
            try:
                json_cell = create_json_cell_vlmr1(
                    sample, 
                    id_prefix, 
                    sample_count,
                    dataset,
                    base_dir
                )
                samples.append(json_cell)
                sample_count += 1
                
            except Exception as e:
                print(f"Warning: Skipping sample {sample_count}: {e}")
                continue

        if sample_count >= num_samples:
            break
    
    return samples


def save_jsonl(samples, output_path):
    """Save samples to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(samples):,} samples to: {output_path}")


def get_valid_combinations():
    """Get all valid task-dataset combinations."""
    combinations = []
    for task, datasets in TASK_DATASET_MAPPING.items():
        for dataset in datasets:
            combinations.append((task, dataset))
    return combinations


def validate_selection(task, dataset):
    """Validate task-dataset combination."""
    if task and dataset:
        # Both specified - check if valid combination
        if task not in TASK_DATASET_MAPPING:
            return False, f"Invalid task '{task}'. Valid tasks: {list(TASK_DATASET_MAPPING.keys())}"
        if dataset not in TASK_DATASET_MAPPING[task]:
            valid_datasets = list(TASK_DATASET_MAPPING[task].keys())
            return False, f"Dataset '{dataset}' not valid for task '{task}'. Valid datasets for {task}: {valid_datasets}"
        return True, "Valid combination"
    
    elif task and not dataset:
        # Only task specified - check if valid task
        if task not in TASK_DATASET_MAPPING:
            return False, f"Invalid task '{task}'. Valid tasks: {list(TASK_DATASET_MAPPING.keys())}"
        return True, "Valid task"
    
    elif not task and dataset:
        # Only dataset specified - check if valid dataset
        if dataset not in DATASET_TASK_MAPPING:
            return False, f"Invalid dataset '{dataset}'. Valid datasets: {list(DATASET_TASK_MAPPING.keys())}"
        return True, "Valid dataset"
    
    else:
        return False, "Must specify either --task, --dataset, or both"


def get_selected_combinations(task, dataset):
    """Get the list of (task, dataset) combinations to process."""
    combinations = []
    
    if task and dataset:
        if dataset.lower() == "both":
            # All datasets for the specified task
            for ds in TASK_DATASET_MAPPING[task]:
                combinations.append((task, ds))
        else:
            # Specific task-dataset combination
            combinations.append((task, dataset))
    
    elif task:
        # All datasets for the specified task
        for ds in TASK_DATASET_MAPPING[task]:
            combinations.append((task, ds))
    
    elif dataset:
        # All tasks that use the specified dataset
        for t in DATASET_TASK_MAPPING[dataset]:
            combinations.append((t, dataset))
    
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="Flexible VLM-R1 dataset generator with task/dataset selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specific task + dataset
  python %(prog)s --task phrase_grounding --dataset MS-CXR
  
  # Task with all its datasets
  python %(prog)s --task phrase_grounding --dataset both
  python %(prog)s --task abnormality_grounding
  
  # Specific dataset (all tasks that use it)
  python %(prog)s --dataset PadChest-GR
  python %(prog)s --dataset VinDr-CXR
  
  # Multiple tasks/datasets in separate runs
  python %(prog)s --task report_generation --dataset MIMIC-CXR
  python %(prog)s --task abnormality_classification --dataset CheXpert

Valid task-dataset combinations:
  report_generation: MIMIC-CXR, CheXpert-Plus
  abnormality_classification: MIMIC-CXR, CheXpert  
  anatomical_grounding: Chest-ImaGenome
  abnormality_grounding: VinDr-CXR
  abnormality_detection: VinDr-CXR
  phrase_grounding: MS-CXR, PadChest-GR
  conversation: MIMIC-CXR
  conversation_grounded: MS-CXR, PadChest-GR
        """
    )
    
    parser.add_argument("--task", choices=list(TASK_DATASET_MAPPING.keys()), 
                       help="Task to generate data for")
    parser.add_argument("--dataset", help="Dataset to use (or 'both' for all datasets in a task)")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", default="./vlmr1_flexible", help="Output directory")
    parser.add_argument("--split", choices=["train", "valid", "test", "both"], default="train", 
                       help="Data split to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--list-combinations", action="store_true", 
                       help="List all valid task-dataset combinations and exit")
    
    args = parser.parse_args()
    
    # List combinations and exit
    if args.list_combinations:
        print("📋 Valid task-dataset combinations:")
        for task, datasets in TASK_DATASET_MAPPING.items():
            print(f"\n🎯 {task}:")
            for dataset, info in datasets.items():
                multiplier_str = f" (×{info['multiplier']})" if info['multiplier'] > 1 else ""
                eval_str = f", eval: {info['eval']}" if info['eval'] else ""
                print(f"  • {dataset}: {info['samples']:,} samples{multiplier_str}{eval_str}")
        return
    
    # Validate selection
    valid, message = validate_selection(args.task, args.dataset)
    if not valid:
        print(f"❌ {message}")
        print("\nUse --list-combinations to see all valid options.")
        return
    
    print(f"🎯 Flexible VLM-R1 Dataset Generator")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Split: {args.split}")
    
    # Get combinations to process
    combinations = get_selected_combinations(args.task, args.dataset)
    print(f"\n🚀 Processing {len(combinations)} task-dataset combination(s):")
    for task, dataset in combinations:
        info = TASK_DATASET_MAPPING[task][dataset]
        multiplier_str = f" (×{info['multiplier']})" if info['multiplier'] > 1 else ""
        print(f"  • {task} + {dataset}: {info['samples']:,} samples{multiplier_str}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    generated_files = []
    
    # Process each combination
    for task, dataset in combinations:
        print(f"\n📁 Processing: {task} + {dataset}")
        
        info = TASK_DATASET_MAPPING[task][dataset]
        multiplier = info['multiplier']
        
        # Process training data
        if args.split in ["train", "both"]:
            try:
                train_dataset = create_dataset(task, dataset, args.data_dir, "train")
                if train_dataset is None:
                    print(f"  ❌ Failed to create training dataset")
                    continue
                
                print(f"  ✅ Created training dataset: {len(train_dataset):,} samples")
                
                # Apply multiplier by generating multiple instances
                all_train_samples = []
                for i in range(multiplier):
                    samples = generate_vlmr1_samples(
                        train_dataset,
                        f"{task}_{dataset}_train_{i+1}",
                        args.data_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=args.seed + i
                    )
                    all_train_samples.extend(samples)
                    print(f"    Instance {i+1}/{multiplier}: {len(samples):,} samples")
                
                # Save training data
                train_filename = f"{task}_{dataset}_train.jsonl".replace("-", "_")
                train_output = os.path.join(args.output_dir, train_filename)
                save_jsonl(all_train_samples, train_output)
                generated_files.append(train_output)
                
            except (PermissionError, FileNotFoundError, Exception) as e:
                print(f"  ❌ Error creating training dataset: {e}")
        
        # Process validation/test data
        if args.split in ["valid", "test", "both"]:
            test_split = "valid" if args.split == "valid" else "test"
            try:
                test_dataset = create_dataset(task, dataset, args.data_dir, test_split)
                if test_dataset is None:
                    print(f"  ℹ️  No {test_split} dataset available")
                else:
                    print(f"  ✅ Created {test_split} dataset: {len(test_dataset):,} samples")
                    
                    test_samples = generate_vlmr1_samples(
                        test_dataset,
                        f"{task}_{dataset}_{test_split}",
                        args.data_dir,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=args.seed
                    )
                    
                    # Save test data
                    test_filename = f"{task}_{dataset}_{test_split}.jsonl".replace("-", "_")
                    test_output = os.path.join(args.output_dir, test_filename)
                    save_jsonl(test_samples, test_output)
                    generated_files.append(test_output)
                    
            except (PermissionError, FileNotFoundError, Exception) as e:
                print(f"  ❌ Error creating {test_split} dataset: {e}")
    
    # Summary
    print(f"\n✅ Generated {len(generated_files)} files:")
    for file_path in generated_files:
        filename = os.path.basename(file_path)
        print(f"  📄 {filename}")
    
    # Usage examples
    if generated_files:
        print(f"\n🚀 Usage with VLM-R1:")
        
        # Single file example
        if len(generated_files) == 1:
            print(f"python -m open_r1.grpo_jsonl \\")
            print(f"    --data_file_paths {generated_files[0]} \\")
            print(f"    --image_folders {args.data_dir}/ \\")
            print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
        
        # Multi-file example
        else:
            train_files = [f for f in generated_files if "_train.jsonl" in f]
            if len(train_files) > 1:
                print(f"# Multi-task training:")
                data_files = ":".join(train_files)
                image_folders = ":".join([args.data_dir + "/"] * len(train_files))
                print(f"python -m open_r1.grpo_jsonl \\")
                print(f"    --data_file_paths {data_files} \\")
                print(f"    --image_folders {image_folders} \\")
                print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")


if __name__ == "__main__":
    main()
