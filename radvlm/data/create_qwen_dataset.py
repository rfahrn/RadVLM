#!/usr/bin/env python3
"""
Improved Qwen-2.5VL dataset generator that handles ALL RadVLM tasks and datasets.

Usage:
    python create_qwen_dataset.py --task report_generation --output_file datasets/qwen_reports.jsonl --split train
    python create_qwen_dataset.py --task abnormality_classification --output_file datasets/qwen_classification.jsonl --split train
    python create_qwen_dataset.py --task region_grounding --output_file datasets/qwen_regions.jsonl --split train
    python create_qwen_dataset.py --task abnormality_grounding --output_file datasets/qwen_abnormalities.jsonl --split train
    python create_qwen_dataset.py --task phrase_grounding --output_file datasets/qwen_phrases.jsonl --split train
    python create_qwen_dataset.py --task conversations --output_file datasets/qwen_conversations.jsonl --split train
"""

import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path

# Add the RadVLM src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from radvlm.data.datasets import (
    MIMIC_Dataset_MM,
    Chest_ImaGenome_Dataset,
    CheXpert_Dataset_MM,
    CheXpertPlus_Dataset,
    VinDr_CXR_Dataset,
    VinDr_CXR_Single_Label_Dataset,
    PadChest_grounding,
    PadChest_grounding_per_image,
    MS_CXR
)

from qwen_dataset_generator import (
    create_json_cell_qwen,
    generate_qwen_dataset_from_instruction_dataset
)


# Task definitions and their corresponding datasets
TASK_CONFIG = {
    "report_generation": {
        "description": "Generate radiology reports from chest X-rays",
        "datasets": ["mimic", "chexpert_plus"],
        "instruction_type": "report"
    },
    "abnormality_classification": {
        "description": "Classify abnormalities present in chest X-rays",
        "datasets": ["mimic", "chexpert", "vindrcxr"],
        "instruction_type": "classification"
    },
    "region_grounding": {
        "description": "Locate anatomical regions in chest X-rays",
        "datasets": ["chest_imagenome", "vindrcxr_single"],
        "instruction_type": "region_location"
    },
    "abnormality_grounding": {
        "description": "Locate abnormalities with bounding boxes",
        "datasets": ["vindrcxr"],
        "instruction_type": "abnormality_location"
    },
    "phrase_grounding": {
        "description": "Locate phrases/sentences in chest X-rays",
        "datasets": ["padchest", "mscxr"],
        "instruction_type": "phrase_location"
    },
    "conversations": {
        "description": "Multi-turn conversations about chest X-rays",
        "datasets": ["mimic_conv", "padchest_conv"],
        "instruction_type": "conversation"
    }
}


def setup_mimic_dataset(data_root, split, task, **kwargs):
    """Setup MIMIC-CXR dataset with appropriate flags for the task."""
    mimic_path = os.path.join(data_root, "mimic-cxr")
    
    if not os.path.exists(mimic_path):
        print(f"MIMIC-CXR path not found: {mimic_path}")
        return None
    
    # Configure flags based on task
    config = {
        "datasetpath": mimic_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "only_frontal": True,
        "seed": kwargs.get("seed", 42)
    }
    
    if task == "report_generation":
        config.update({
            "flag_txt": True,
            "flag_lab": False,
            "classif": False,
            "filtered_reports_dir": kwargs.get("filtered_reports_dir")
        })
    elif task == "abnormality_classification":
        config.update({
            "flag_txt": False,
            "flag_lab": True,
            "classif": True
        })
    elif task == "conversations":
        config.update({
            "flag_txt": True,
            "flag_lab": False,
            "conversation_dir": kwargs.get("conversation_dir")
        })
    
    return MIMIC_Dataset_MM(**config)


def setup_chest_imagenome_dataset(data_root, split, task, **kwargs):
    """Setup Chest ImaGenome dataset for region grounding."""
    chest_ima_path = os.path.join(data_root, "chest-imagenome")
    mimic_path = os.path.join(data_root, "mimic-cxr")
    
    if not os.path.exists(chest_ima_path) or not os.path.exists(mimic_path):
        print(f"Chest ImaGenome or MIMIC path not found")
        return None
    
    config = {
        "datasetpath_chestima": chest_ima_path,
        "datasetpath": mimic_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "flag_txt": False,
        "flag_lab": False,
        "only_frontal": True,
        "pick_one_region": True,
        "seed": kwargs.get("seed", 42)
    }
    
    return Chest_ImaGenome_Dataset(**config)


def setup_chexpert_dataset(data_root, split, task, **kwargs):
    """Setup CheXpert dataset for classification."""
    chexpert_path = os.path.join(data_root, "chexpert")
    
    if not os.path.exists(chexpert_path):
        print(f"CheXpert path not found: {chexpert_path}")
        return None
    
    config = {
        "datasetpath": chexpert_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "flag_lab": True,
        "only_frontal": True,
        "seed": kwargs.get("seed", 42)
    }
    
    return CheXpert_Dataset_MM(**config)


def setup_chexpert_plus_dataset(data_root, split, task, **kwargs):
    """Setup CheXpert Plus dataset for report generation."""
    chexpert_plus_path = os.path.join(data_root, "chexpert-plus")
    
    if not os.path.exists(chexpert_plus_path):
        print(f"CheXpert Plus path not found: {chexpert_plus_path}")
        return None
    
    config = {
        "datasetpath": chexpert_plus_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "flag_txt": True,
        "flag_lab": False,
        "only_frontal": True,
        "filtered_reports_dir": kwargs.get("filtered_reports_dir"),
        "seed": kwargs.get("seed", 42)
    }
    
    return CheXpertPlus_Dataset(**config)


def setup_vindrcxr_dataset(data_root, split, task, **kwargs):
    """Setup VinDr-CXR dataset for abnormality grounding."""
    vindrcxr_path = os.path.join(data_root, "vindrcxr")
    
    if not os.path.exists(vindrcxr_path):
        print(f"VinDr-CXR path not found: {vindrcxr_path}")
        return None
    
    config = {
        "datasetpath": vindrcxr_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "seed": kwargs.get("seed", 42)
    }
    
    if task == "region_grounding":
        return VinDr_CXR_Single_Label_Dataset(**config)
    else:
        return VinDr_CXR_Dataset(**config)


def setup_padchest_dataset(data_root, split, task, **kwargs):
    """Setup PadChest dataset for phrase grounding."""
    padchest_path = os.path.join(data_root, "padchest")
    
    if not os.path.exists(padchest_path):
        print(f"PadChest path not found: {padchest_path}")
        return None
    
    config = {
        "datasetpath": padchest_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "flag_txt": False,
        "seed": kwargs.get("seed", 42)
    }
    
    if task == "conversations":
        config["conversation_dir"] = kwargs.get("conversation_dir")
        return PadChest_grounding_per_image(**config)
    else:
        return PadChest_grounding(**config)


def setup_mscxr_dataset(data_root, split, task, **kwargs):
    """Setup MS-CXR dataset for phrase grounding."""
    mimic_path = os.path.join(data_root, "mimic-cxr")
    
    if not os.path.exists(mimic_path):
        print(f"MIMIC-CXR path not found for MS-CXR: {mimic_path}")
        return None
    
    config = {
        "datasetpath": mimic_path,
        "split": split,
        "flag_img": True,
        "flag_instr": True,
        "flag_txt": False,
        "flag_lab": False,
        "only_frontal": True,
        "sentencesBBoxpath": kwargs.get("sentencesBBoxpath"),
        "seed": kwargs.get("seed", 42)
    }
    
    return MS_CXR(**config)


def setup_datasets_for_task(task, data_root, split, **kwargs):
    """Setup all relevant datasets for a given task."""
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(TASK_CONFIG.keys())}")
    
    datasets = []
    task_datasets = TASK_CONFIG[task]["datasets"]
    
    for dataset_name in task_datasets:
        dataset = None
        id_prefix = f"{dataset_name}_{split}"
        
        if dataset_name == "mimic":
            dataset = setup_mimic_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "mimic_conv":
            dataset = setup_mimic_dataset(data_root, split, "conversations", **kwargs)
            id_prefix = f"mimic_conv_{split}"
        elif dataset_name == "chest_imagenome":
            dataset = setup_chest_imagenome_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "chexpert":
            dataset = setup_chexpert_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "chexpert_plus":
            dataset = setup_chexpert_plus_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "vindrcxr":
            dataset = setup_vindrcxr_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "vindrcxr_single":
            dataset = setup_vindrcxr_dataset(data_root, split, "region_grounding", **kwargs)
            id_prefix = f"vindrcxr_single_{split}"
        elif dataset_name == "padchest":
            dataset = setup_padchest_dataset(data_root, split, task, **kwargs)
        elif dataset_name == "padchest_conv":
            dataset = setup_padchest_dataset(data_root, split, "conversations", **kwargs)
            id_prefix = f"padchest_conv_{split}"
        elif dataset_name == "mscxr":
            dataset = setup_mscxr_dataset(data_root, split, task, **kwargs)
        
        if dataset is not None and len(dataset) > 0:
            datasets.append({
                "dataset": dataset,
                "id_prefix": id_prefix,
                "num_samples": kwargs.get("num_samples")
            })
            print(f"✅ Loaded {id_prefix}: {len(dataset)} samples")
        else:
            print(f"❌ Failed to load or empty dataset: {dataset_name}")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qwen-2.5VL format datasets for RadVLM tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks:
{chr(10).join([f"  {task}: {config['description']}" for task, config in TASK_CONFIG.items()])}

Example usage:
  python create_qwen_dataset.py --task report_generation --output_file datasets/qwen_reports.jsonl --split train
  python create_qwen_dataset.py --task abnormality_classification --output_file datasets/qwen_classification.jsonl --split train --data_root /path/to/datasets
        """
    )
    
    parser.add_argument("--task", type=str, required=True, 
                        choices=list(TASK_CONFIG.keys()),
                        help="Task type to generate dataset for")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--split", type=str, choices=["train", "valid", "test"], default="train",
                        help="Dataset split to process")
    parser.add_argument("--data_root", type=str, default="/path/to/datasets",
                        help="Root directory containing all datasets")
    
    # Dataset-specific arguments
    parser.add_argument("--filtered_reports_dir", type=str, default=None,
                        help="Directory with filtered reports (for MIMIC/CheXpert Plus)")
    parser.add_argument("--conversation_dir", type=str, default=None,
                        help="Directory with conversation files")
    parser.add_argument("--sentencesBBoxpath", type=str, default=None,
                        help="Directory with sentence bounding box files (for MS-CXR)")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples per dataset (None = all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"🎯 Task: {args.task}")
    print(f"📁 Output: {args.output_file}")
    print(f"📊 Split: {args.split}")
    print(f"🗂️ Data root: {args.data_root}")
    print(f"🎲 Seed: {args.seed}")
    print()
    
    # Setup datasets for the specified task
    datasets = setup_datasets_for_task(
        task=args.task,
        data_root=args.data_root,
        split=args.split,
        filtered_reports_dir=args.filtered_reports_dir,
        conversation_dir=args.conversation_dir,
        sentencesBBoxpath=args.sentencesBBoxpath,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    if not datasets:
        print("❌ No datasets loaded! Please check your data paths and configuration.")
        return
    
    print(f"\n📋 Loaded {len(datasets)} dataset(s) for task '{args.task}'")
    total_samples = sum(len(d["dataset"]) for d in datasets)
    print(f"📊 Total samples available: {total_samples}")
    
    # Generate the Qwen-2.5VL format dataset
    print(f"\n🔄 Generating Qwen-2.5VL dataset...")
    generated_samples = generate_qwen_dataset_from_instruction_dataset(
        dataset_info=datasets,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"\n✅ Successfully generated {generated_samples} samples!")
    print(f"📁 Dataset saved to: {args.output_file}")
    
    # Show task summary
    print(f"\n📋 Task Summary:")
    print(f"   Task: {args.task}")
    print(f"   Description: {TASK_CONFIG[args.task]['description']}")
    print(f"   Instruction type: {TASK_CONFIG[args.task]['instruction_type']}")
    print(f"   Generated samples: {generated_samples}")
    
    # Validate the output format
    print(f"\n🔍 Validating output format...")
    validate_qwen_format(args.output_file)


def validate_qwen_format(jsonl_file, num_samples_to_check=3):
    """Validate that the generated JSONL follows Qwen-2.5VL format."""
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples_to_check:
                    break
                    
                data = json.loads(line.strip())
                
                # Check required fields
                assert "id" in data, f"Missing 'id' field in line {i+1}"
                assert "image" in data, f"Missing 'image' field in line {i+1}" 
                assert "conversations" in data, f"Missing 'conversations' field in line {i+1}"
                assert isinstance(data["id"], int), f"ID should be integer in line {i+1}"
                assert isinstance(data["conversations"], list), f"Conversations should be list in line {i+1}"
                
                # Check conversation format
                for conv in data["conversations"]:
                    assert "from" in conv, f"Missing 'from' in conversation in line {i+1}"
                    assert "value" in conv, f"Missing 'value' in conversation in line {i+1}"
                    assert conv["from"] in ["human", "gpt"], f"Invalid 'from' value in line {i+1}"
                
                # Check that first human message contains <image> token
                human_messages = [c for c in data["conversations"] if c["from"] == "human"]
                if human_messages:
                    assert "<image>" in human_messages[0]["value"], f"Missing <image> token in first human message in line {i+1}"
                
                print(f"✓ Line {i+1}: Valid format - Task sample")
        
        print(f"✅ Format validation successful! Checked first {num_samples_to_check} samples.")
        
        # Show a sample
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            sample = json.loads(f.readline().strip())
            print(f"\n📄 Sample entry:")
            print(f"   ID: {sample['id']}")
            print(f"   Image: {sample['image']}")
            print(f"   Conversations: {len(sample['conversations'])} turns")
            if sample['conversations']:
                print(f"   First turn: {sample['conversations'][0]['value'][:100]}...")
        
    except Exception as e:
        print(f"❌ Format validation failed: {e}")


if __name__ == "__main__":
    main()
