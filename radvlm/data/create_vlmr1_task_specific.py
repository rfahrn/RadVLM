#!/usr/bin/env python
"""
Task-specific VLM-R1 dataset generator for RadVLM.
Allows generation of specific task+dataset combinations with --task and --dataset parameters.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from radvlm.data.datasets import (
    MIMIC_Dataset_MM, CheXpert_Dataset_MM, Chest_ImaGenome_Dataset, 
    MS_CXR, CheXpertPlus_Dataset, PadChest_grounding, 
    PadChest_grounding_per_image, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset
)
from radvlm.data.utils import custom_collate_fn

# Import helper functions from the comprehensive script
from create_vlmr1_comprehensive import (
    convert_conversation_to_vlmr1, 
    convert_instruction_to_vlmr1,
    make_image_path_relative,
    create_vlmr1_sample
)


# Task and dataset mappings
TASK_DATASET_CONFIGS = {
    "report_generation": {
        "mimic_cxr": {
            "name": "mimic_reports",
            "description": "MIMIC-CXR Report Generation",
            "expected_samples": 230980,
            "config_func": "get_mimic_reports_config"
        },
        "chexpert_plus": {
            "name": "chexpert_plus_reports", 
            "description": "CheXpert-Plus Report Generation",
            "expected_samples": 186463,
            "config_func": "get_chexpert_plus_config"
        }
    },
    "abnormality_classification": {
        "mimic_cxr": {
            "name": "mimic_classification",
            "description": "MIMIC-CXR Abnormality Classification", 
            "expected_samples": 237912,
            "config_func": "get_mimic_classification_config"
        },
        "chexpert": {
            "name": "chexpert_classification",
            "description": "CheXpert Abnormality Classification",
            "expected_samples": 191027,
            "config_func": "get_chexpert_config"
        }
    },
    "anatomical_grounding": {
        "chest_imagenome": {
            "name": "chest_imagenome",
            "description": "Chest ImaGenome Anatomical Grounding",
            "expected_samples": 80000,
            "config_func": "get_chest_imagenome_config"
        }
    },
    "abnormality_grounding": {
        "vindr_cxr": {
            "name": "vindr_abnormality_grounding",
            "description": "VinDr-CXR Abnormality Grounding",
            "expected_samples": 16089,
            "config_func": "get_vindr_grounding_config"
        }
    },
    "abnormality_detection": {
        "vindr_cxr": {
            "name": "vindr_detection",
            "description": "VinDr-CXR Abnormality Detection",
            "expected_samples": 15000,
            "config_func": "get_vindr_detection_config"
        }
    },
    "phrase_grounding": {
        "ms_cxr": {
            "name": "ms_cxr_grounding",
            "description": "MS-CXR Phrase Grounding",
            "expected_samples": 971,
            "config_func": "get_ms_cxr_config"
        },
        "padchest": {
            "name": "padchest_grounding",
            "description": "PadChest Phrase Grounding",
            "expected_samples": 4478,
            "config_func": "get_padchest_grounding_config"
        }
    },
    "conversation": {
        "mimic_cxr": {
            "name": "mimic_conversation",
            "description": "MIMIC-CXR Conversations",
            "expected_samples": 86155,
            "config_func": "get_mimic_conversation_config"
        }
    },
    "conversation_grounded": {
        "ms_cxr": {
            "name": "ms_cxr_conversation",
            "description": "MS-CXR Grounded Conversations",
            "expected_samples": 862,
            "config_func": "get_ms_cxr_conversation_config"
        },
        "padchest": {
            "name": "padchest_conversation",
            "description": "PadChest Grounded Conversations", 
            "expected_samples": 2225,
            "config_func": "get_padchest_conversation_config"
        }
    }
}


def get_mimic_reports_config(data_dir, split="train"):
    """MIMIC-CXR Report Generation dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    
    return {
        "name": f"mimic_reports_{split}",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=split, 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        ),
        "id_prefix": f"mimic-reports-{split}"
    }


def get_chexpert_plus_config(data_dir, split="train"):
    """CheXpert-Plus Report Generation dataset config."""
    datasetpath = os.path.join(data_dir, 'CheXpert')
    filtered_reports_dir = os.path.join(datasetpath, 'filtered_reports')
    
    return {
        "name": f"chexpert_plus_{split}",
        "dataset": CheXpertPlus_Dataset(
            datasetpath=datasetpath, 
            split=split, 
            flag_img=False, 
            filtered_reports_dir=filtered_reports_dir
        ),
        "id_prefix": f"chexpert-plus-{split}"
    }


def get_mimic_classification_config(data_dir, split="train"):
    """MIMIC-CXR Classification dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    
    return {
        "name": f"mimic_classification_{split}",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=split, 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        ),
        "id_prefix": f"mimic-classif-{split}"
    }


def get_chexpert_config(data_dir, split="train"):
    """CheXpert Classification dataset config."""
    dataset_path = os.path.join(data_dir, "CheXpert")   
    
    return {
        "name": f"chexpert_{split}",
        "dataset": CheXpert_Dataset_MM(
            datasetpath=dataset_path,
            split=split, 
            flag_img=False
        ),
        "id_prefix": f"chexpert-{split}"
    }


def get_chest_imagenome_config(data_dir, split="train"):
    """Chest ImaGenome Anatomical Grounding dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
    
    return {
        "name": f"chest_imagenome_{split}",
        "dataset": Chest_ImaGenome_Dataset(
            datasetpath=datasetpath_mimic,
            datasetpath_chestima=datasetpath_chestima,
            split=split, 
            flag_img=False, 
            flag_instr=True, 
            flag_txt=False, 
            flag_lab=False,
            pick_one_region=True,
        ),
        "id_prefix": f"chestima-{split}",
        "num_samples": 80000
    }


def get_vindr_grounding_config(data_dir, split="train"):
    """VinDr-CXR Abnormality Grounding dataset config."""
    dataset_path = os.path.join(data_dir, "VinDr-CXR") 
    
    return {
        "name": f"vindr_grounding_{split}",
        "dataset": VinDr_CXR_Dataset(
            datasetpath=dataset_path, 
            split=split, 
            flag_img=False
        ),
        "id_prefix": f"vindr-grounding-{split}"
    }


def get_vindr_detection_config(data_dir, split="train"):
    """VinDr-CXR Abnormality Detection dataset config."""
    dataset_path = os.path.join(data_dir, "VinDr-CXR")
    
    return {
        "name": f"vindr_detection_{split}",
        "dataset": VinDr_CXR_Single_Label_Dataset(
            datasetpath=dataset_path, 
            split=split, 
            flag_img=False
        ),
        "id_prefix": f"vindr-detection-{split}"
    }


def get_ms_cxr_config(data_dir, split="train"):
    """MS-CXR Phrase Grounding dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    
    dataset_train = MS_CXR(
        datasetpath=datasetpath_mimic,
        split="train", flag_img=False, 
        flag_lab=True, only_frontal=True, 
        flag_instr=True, 
        sentencesBBoxpath=sentencesBBoxpath,
        seed=0
    )
    
    if split == "train":
        return {
            "name": f"ms_cxr_{split}",
            "dataset": dataset_train,
            "id_prefix": f"mscxr-{split}"
        }
    else:
        dataset_valid = MS_CXR(
            datasetpath=datasetpath_mimic,
            split="valid", flag_img=False, 
            flag_lab=True, only_frontal=True, 
            flag_instr=True, 
            sentencesBBoxpath=sentencesBBoxpath,
            seed=0
        )
        return {
            "name": f"ms_cxr_{split}",
            "dataset": dataset_valid,
            "id_prefix": f"mscxr-{split}"
        }


def get_padchest_grounding_config(data_dir, split="train"):
    """PadChest Phrase Grounding dataset config."""
    datasetpath_pad = os.path.join(data_dir, 'PadChest')
    
    return {
        "name": f"padchest_grounding_{split}",
        "dataset": PadChest_grounding(
            datasetpath=datasetpath_pad,
            split=split, 
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        ),
        "id_prefix": f"padchest-grounding-{split}"
    }


def get_mimic_conversation_config(data_dir, split="train"):
    """MIMIC-CXR Conversations dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    conversation_dir = os.path.join(datasetpath_mimic, f'conversations/{split}/standard')
    
    if not os.path.exists(conversation_dir):
        raise ValueError(f"Conversation directory not found: {conversation_dir}")
    
    return {
        "name": f"mimic_conversation_{split}",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=split, 
            flag_img=False, 
            flag_instr=False, 
            flag_txt=False, 
            flag_lab=False, 
            filtered_reports_dir=filtered_reports_dir,
            conversation_dir=conversation_dir
        ),
        "id_prefix": f"mimic-conv-{split}"
    }


def get_ms_cxr_conversation_config(data_dir, split="train"):
    """MS-CXR Grounded Conversations dataset config."""
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    conversation_dir_grounded = os.path.join(datasetpath_mimic, f'conversations/{split}/grounding')
    
    if not os.path.exists(conversation_dir_grounded):
        raise ValueError(f"Grounded conversation directory not found: {conversation_dir_grounded}")
    
    return {
        "name": f"ms_cxr_conversation_{split}",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=split, flag_img=False, 
            flag_lab=False, only_frontal=True, 
            flag_instr=False, 
            filtered_reports_dir=filtered_reports_dir,
            sentencesBBoxpath=sentencesBBoxpath,
            conversation_dir=conversation_dir_grounded,
            classif=False,
            seed=0
        ),
        "id_prefix": f"mscxr-conv-{split}"
    }


def get_padchest_conversation_config(data_dir, split="train"):
    """PadChest Grounded Conversations dataset config."""
    datasetpath_pad = os.path.join(data_dir, 'PadChest')
    conversation_dir_pad = os.path.join(datasetpath_pad, f'conversations/{split}/grounding')
    
    if not os.path.exists(conversation_dir_pad):
        raise ValueError(f"PadChest conversation directory not found: {conversation_dir_pad}")
    
    return {
        "name": f"padchest_conversation_{split}",
        "dataset": PadChest_grounding_per_image(
            datasetpath=datasetpath_pad,
            split=split,
            flag_instr=False,
            flag_img=False,
            conversation_dir=conversation_dir_pad
        ),
        "id_prefix": f"padchest-conv-{split}"
    }


def get_dataset_config(task, dataset, data_dir, split="train"):
    """Get dataset configuration for specific task and dataset."""
    if task not in TASK_DATASET_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available tasks: {list(TASK_DATASET_CONFIGS.keys())}")
    
    if dataset not in TASK_DATASET_CONFIGS[task]:
        available_datasets = list(TASK_DATASET_CONFIGS[task].keys())
        raise ValueError(f"Dataset '{dataset}' not available for task '{task}'. Available: {available_datasets}")
    
    config_info = TASK_DATASET_CONFIGS[task][dataset]
    config_func_name = config_info["config_func"]
    config_func = globals()[config_func_name]
    
    try:
        return config_func(data_dir, split)
    except Exception as e:
        print(f"Error creating dataset config for {task}/{dataset}: {e}")
        raise


def list_available_tasks_datasets():
    """Print all available task and dataset combinations."""
    print("\nAvailable Task-Dataset Combinations:")
    print("=" * 50)
    
    for task, datasets in TASK_DATASET_CONFIGS.items():
        print(f"\nTask: {task}")
        for dataset_name, info in datasets.items():
            print(f"  └─ {dataset_name}: {info['description']} (~{info['expected_samples']:,} samples)")
    
    print("\nUsage examples:")
    print("python create_vlmr1_task_specific.py --task report_generation --dataset mimic_cxr")
    print("python create_vlmr1_task_specific.py --task phrase_grounding --dataset padchest")
    print("python create_vlmr1_task_specific.py --task abnormality_classification --dataset chexpert")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM-R1 compatible JSONL for specific task+dataset combinations"
    )
    parser.add_argument(
        "--task", 
        required=False,
        choices=list(TASK_DATASET_CONFIGS.keys()),
        help="Task type to generate"
    )
    parser.add_argument(
        "--dataset",
        required=False, 
        help="Dataset to use for the task"
    )
    parser.add_argument(
        "--data-dir", 
        required=True,
        help="Root data directory containing all datasets"
    )
    parser.add_argument(
        "--output-dir", 
        default="./vlmr1_task_datasets",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
        help="Data split to generate"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=8,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--shuffle", 
        action="store_true",
        help="Shuffle the datasets"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available task-dataset combinations and exit"
    )
    
    args = parser.parse_args()
    
    # List available tasks and exit
    if args.list_tasks:
        list_available_tasks_datasets()
        return 0
    
    # Validate required arguments
    if not args.task:
        print("❌ Error: --task is required. Use --list-tasks to see available options.")
        return 1
    
    if not args.dataset:
        available_datasets = list(TASK_DATASET_CONFIGS[args.task].keys())
        print(f"❌ Error: --dataset is required for task '{args.task}'.")
        print(f"Available datasets: {available_datasets}")
        return 1
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating VLM-R1 dataset for:")
    print(f"  Task: {args.task}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    
    try:
        # Get dataset configuration
        config = get_dataset_config(args.task, args.dataset, args.data_dir, args.split)
        dataset = config["dataset"]
        
        print(f"\nDataset loaded: {len(dataset):,} samples")
        
        # Determine number of samples to process
        num_samples = config.get("num_samples", len(dataset))
        if args.limit:
            num_samples = min(num_samples, args.limit)
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        
        dataset_samples = []
        sample_count = 0
        
        print(f"Processing {num_samples:,} samples...")
        
        for batch in data_loader:
            for sample in batch:
                if sample_count >= num_samples:
                    break
                
                try:
                    vlmr1_sample = create_vlmr1_sample(
                        sample, 
                        sample_count,
                        args.data_dir,
                        config["id_prefix"]
                    )
                    dataset_samples.append(vlmr1_sample)
                    sample_count += 1
                    
                    if sample_count % 1000 == 0:
                        print(f"  Processed {sample_count:,} samples...")
                    
                except Exception as e:
                    print(f"Warning: Skipping sample {sample_count}: {e}")
                    continue
            
            if sample_count >= num_samples:
                break
        
        # Save output file
        output_filename = f"{args.task}_{args.dataset}_{args.split}.jsonl"
        output_file = os.path.join(args.output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Successfully generated {len(dataset_samples):,} samples")
        print(f"📁 Saved to: {output_file}")
        
        # Show usage example for VLM-R1
        print(f"\n🚀 Usage with VLM-R1:")
        print(f"python -m open_r1.grpo_jsonl \\")
        print(f"    --data_file_paths {output_file} \\")
        print(f"    --image_folders {args.data_dir}/ \\")
        print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())