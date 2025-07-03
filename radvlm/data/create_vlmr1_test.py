#!/usr/bin/env python
"""
Script to generate VLM-R1 compatible test/validation JSONL datasets from RadVLM.
This complements the training dataset generation script.
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
from radvlm import DATA_DIR

# Import helper functions from the main script
from create_vlmr1_comprehensive import (
    convert_conversation_to_vlmr1, 
    convert_instruction_to_vlmr1,
    make_image_path_relative,
    create_vlmr1_sample
)


def get_test_dataset_configs(data_dir):
    """
    Get test/validation dataset configurations.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        List of dataset configuration dicts for test/validation
    """
    # MIMIC-CXR paths
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    
    datasets = []
    
    # MIMIC-CXR validation reports
    datasets.append({
        "name": "mimic_reports_test",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="valid", 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        ),
        "id_prefix": "mimic-test"
    })
    
    # MIMIC-CXR validation classification
    datasets.append({
        "name": "mimic_labels_test",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="valid", 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        ),
        "id_prefix": "mimic-labels-test"
    })
    
    # CheXpert validation
    dataset_path = os.path.join(data_dir, "CheXpert")   
    datasets.append({
        "name": "chexpert_test",
        "dataset": CheXpert_Dataset_MM(
            datasetpath=dataset_path,
            split="valid", 
            flag_img=False
        ),
        "id_prefix": "chexpert-test"
    })
    
    # VinDr-CXR test
    dataset_path = os.path.join(data_dir, "VinDr-CXR")
    if os.path.exists(dataset_path):
        datasets.append({
            "name": "vindr_cxr_test",
            "dataset": VinDr_CXR_Dataset(
                datasetpath=dataset_path, 
                split="test", 
                flag_img=False
            ),
            "id_prefix": "vindr-cxr-test"
        })
        
        datasets.append({
            "name": "vindr_cxr_mono_test",
            "dataset": VinDr_CXR_Single_Label_Dataset(
                datasetpath=dataset_path, 
                split="test", 
                flag_img=False
            ),
            "id_prefix": "vindr-cxr-mono-test"
        })
    
    # PadChest test phrase grounding (using 'test' split if available)
    datasetpath_pad = os.path.join(data_dir, 'PadChest')
    if os.path.exists(datasetpath_pad):
        try:
            dataset_test_pad = PadChest_grounding(
                datasetpath=datasetpath_pad,
                split="test", 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
            datasets.append({
                "name": "padchest_grounding_test",
                "dataset": dataset_test_pad,
                "id_prefix": "padchest-test"
            })
        except:
            # If no test split, use validation split
            dataset_valid_pad = PadChest_grounding(
                datasetpath=datasetpath_pad,
                split="valid", 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
            datasets.append({
                "name": "padchest_grounding_test",
                "dataset": dataset_valid_pad,
                "id_prefix": "padchest-test"
            })
    
    # Conversations validation (if available)
    conversation_dir = os.path.join(datasetpath_mimic, 'conversations/valid/standard')
    if os.path.exists(conversation_dir):
        datasets.append({
            "name": "conversations_standard_test",
            "dataset": MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="valid", 
                flag_img=False, 
                flag_instr=False, 
                flag_txt=False, 
                flag_lab=False, 
                filtered_reports_dir=filtered_reports_dir,
                conversation_dir=conversation_dir
            ),
            "id_prefix": "conv-test"
        })
    
    # Conversations grounded validation
    conversation_dir_grounded = os.path.join(datasetpath_mimic, 'conversations/valid/grounding')
    if os.path.exists(conversation_dir_grounded):
        datasets.append({
            "name": "conversations_grounded_test",
            "dataset": MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="valid", flag_img=False, 
                flag_lab=False, only_frontal=True, 
                flag_instr=False, 
                filtered_reports_dir=filtered_reports_dir,
                sentencesBBoxpath=sentencesBBoxpath,
                conversation_dir=conversation_dir_grounded,
                classif=False,
                seed=0
            ),
            "id_prefix": "conv-grounded-test"
        })
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM-R1 compatible test/validation JSONL datasets from RadVLM"
    )
    parser.add_argument(
        "--data-dir", 
        default=DATA_DIR,
        help="Root data directory containing all datasets"
    )
    parser.add_argument(
        "--output-dir", 
        default="./vlmr1_datasets",
        help="Output directory for JSONL files"
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
        "--combine-all", 
        action="store_true",
        help="Create a single combined JSONL file with all test datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset names to process (if not specified, processes all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset configurations
    dataset_configs = get_test_dataset_configs(args.data_dir)
    
    # Filter datasets if specified
    if args.datasets:
        dataset_configs = [cfg for cfg in dataset_configs if cfg["name"] in args.datasets]
    
    print(f"Processing {len(dataset_configs)} test datasets...")
    
    all_samples = []
    
    for config in dataset_configs:
        print(f"\nProcessing {config['name']} ({config['id_prefix']})...")
        
        dataset = config["dataset"]
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
                    all_samples.append(vlmr1_sample)
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Warning: Skipping sample {sample_count} in {config['name']}: {e}")
                    continue
            
            if sample_count >= num_samples:
                break
        
        print(f"Processed {len(dataset_samples)} samples from {config['name']}")
        
        # Save individual dataset file if not combining all
        if not args.combine_all:
            output_file = os.path.join(args.output_dir, f"{config['name']}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in dataset_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Saved {len(dataset_samples)} samples to {output_file}")
    
    # Save combined file if requested
    if args.combine_all:
        output_file = os.path.join(args.output_dir, "all_test.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(all_samples)} total samples to {output_file}")
    
    print(f"\n✅ Test dataset processing complete! Generated {len(all_samples)} total samples.")
    print(f"Files saved to: {args.output_dir}")
    print(f"\nUsage with VLM-R1:")
    print(f"--data_file_paths {args.output_dir}/all_test.jsonl")
    print(f"--image_folders {args.data_dir}/")


if __name__ == "__main__":
    main()