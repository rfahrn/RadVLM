import torch
import numpy as np
from radvlm.data.datasets import (
    MS_CXR, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset,
    PadChest_grounding, PadChest_grounding_per_image, MIMIC_Dataset_MM
)
from radvlm.data.utils import custom_collate_fn
import json
import os
from torch.utils.data import ConcatDataset, DataLoader
from radvlm import DATA_DIR
import pandas as pd
from tqdm import tqdm
import random
import argparse
from sklearn.model_selection import train_test_split

def extract_iou_solution(boxes):
    """
    Extract IoU solution from bounding boxes for grounding tasks.
    Returns normalized bounding boxes in format expected by veRL.
    """
    if not boxes:
        return []
    
    # Ensure boxes are in [x1, y1, x2, y2] format (normalized)
    processed_boxes = []
    for box in boxes:
        if isinstance(box, list) and len(box) == 4:
            # Ensure coordinates are normalized (0-1 range)
            x1, y1, x2, y2 = box
            processed_boxes.append([float(x1), float(y1), float(x2), float(y2)])
    
    return processed_boxes

def format_prompt_for_verl(sample, instruction_template=None):
    """
    Formats the instruction or conversation from a RadVLM sample into
    the list-of-dictionaries format required by VeRL's chat template.
    """
    # Default instruction for grounding tasks
    if instruction_template is None:
        instruction_template = "Please provide the bounding box coordinates for the following description: {}"
    
    # Handle different sample formats
    if 'instr' in sample and sample['instr']:
        instruction_data = sample['instr']
    elif 'conversation' in sample and sample['conversation']:
        instruction_data = sample['conversation']
    elif 'label' in sample and sample['label']:
        # Create instruction from label
        question = instruction_template.format(sample['label'])
        instruction_data = {
            'question': question,
            'answer': f"The bounding box coordinates are: {sample.get('boxes', [])}"
        }
    else:
        return None

    if isinstance(instruction_data, dict):
        question = instruction_data.get('question', '')
        verl_prompt = [{'role': 'user', 'content': question}]
        
        # For grounding tasks, we don't include the answer in the prompt
        # The model will generate it, and we'll use IoU for reward
        
    elif isinstance(instruction_data, list):
        verl_prompt = []
        for turn in instruction_data:
            if not isinstance(turn, dict) or 'from' not in turn or 'value' not in turn:
                continue
            role = 'user' if turn['from'] == 'human' else 'assistant'
            content = turn['value'].replace('<image>\n', '').replace('<image>', '').strip()
            verl_prompt.append({'role': role, 'content': content})
    else:
        return None

    return verl_prompt

def create_grounding_dataset(dataset_configs, split='train', batch_size=32, num_workers=4, seed=42):
    """
    Create a veRL-compatible grounding dataset from multiple dataset configurations.
    """
    verl_samples = []
    np.random.seed(seed)
    random.seed(seed)
    
    for config in dataset_configs:
        dataset = config["dataset"]
        dataset_name = config["name"]
        max_samples = config.get("max_samples", len(dataset))
        
        print(f"Processing {dataset_name} ({split}) - {max_samples} samples")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        sample_count = 0
        progress_bar = tqdm(total=max_samples, desc=f"{dataset_name} ({split})")
        
        for batch in data_loader:
            for sample in batch:
                if sample_count >= max_samples:
                    break
                
                # Extract grounding truth
                gt_boxes = sample.get('boxes', [])
                if not gt_boxes:
                    continue
                
                # Extract label/phrase
                label = sample.get('label', sample.get('observation', ''))
                if not label:
                    continue
                
                # Format prompt for veRL
                verl_prompt = format_prompt_for_verl(sample)
                if not verl_prompt:
                    continue
                
                # Create veRL sample
                verl_sample = {
                    "data_source": dataset_name,
                    "prompt": verl_prompt,
                    "ability": "phrase_grounding",
                    "reward_model": {
                        "style": "iou",
                        "ground_truth": extract_iou_solution(gt_boxes),
                        "label": label
                    },
                    "extra_info": {
                        "image_path": sample.get('img_path', ''),
                        "dataset_source": dataset_name,
                        "split": split,
                        "sample_id": f"{dataset_name}_{split}_{sample_count}"
                    }
                }
                
                verl_samples.append(verl_sample)
                sample_count += 1
                progress_bar.update(1)
                
            if sample_count >= max_samples:
                break
                
        progress_bar.close()
        print(f"Completed {dataset_name} ({split}): {sample_count} samples")
    
    return verl_samples

def setup_datasets(data_dir):
    """
    Setup all grounding datasets for train and test splits.
    """
    # MIMIC-CXR paths
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR', 'sentences_and_BBox_mscxr')
    
    # VinDr-CXR paths
    dataset_path_vindr = os.path.join(data_dir, "VinDr-CXR")
    
    # PadChest paths
    datasetpath_padchest = os.path.join(data_dir, 'PadChest')
    
    # Initialize datasets
    datasets = {}
    
    # MS-CXR (phrase grounding)
    if os.path.exists(sentencesBBoxpath):
        datasets['ms_cxr_train'] = MS_CXR(
            datasetpath=datasetpath_mimic,
            split="train",
            flag_img=False,
            flag_lab=True,
            only_frontal=True,
            flag_instr=True,
            sentencesBBoxpath=sentencesBBoxpath,
            seed=42
        )
        datasets['ms_cxr_test'] = MS_CXR(
            datasetpath=datasetpath_mimic,
            split="test",
            flag_img=False,
            flag_lab=True,
            only_frontal=True,
            flag_instr=True,
            sentencesBBoxpath=sentencesBBoxpath,
            seed=42
        )
    
    # VinDr-CXR (abnormality grounding)
    if os.path.exists(dataset_path_vindr):
        datasets['vindr_train'] = VinDr_CXR_Dataset(
            datasetpath=dataset_path_vindr,
            split="train",
            flag_img=False
        )
        datasets['vindr_test'] = VinDr_CXR_Dataset(
            datasetpath=dataset_path_vindr,
            split="test",
            flag_img=False
        )
        
        # VinDr single label version
        datasets['vindr_single_train'] = VinDr_CXR_Single_Label_Dataset(
            datasetpath=dataset_path_vindr,
            split="train",
            flag_img=False
        )
        datasets['vindr_single_test'] = VinDr_CXR_Single_Label_Dataset(
            datasetpath=dataset_path_vindr,
            split="test",
            flag_img=False
        )
    
    # PadChest (grounding)
    if os.path.exists(datasetpath_padchest):
        datasets['padchest_train'] = PadChest_grounding(
            datasetpath=datasetpath_padchest,
            split='train',
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )
        datasets['padchest_test'] = PadChest_grounding(
            datasetpath=datasetpath_padchest,
            split='test',
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )
    
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Create comprehensive grounding dataset for veRL GRPO training")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='./grounding_datasets', help='Output directory for parquet files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max_samples_per_dataset', type=int, default=None, help='Maximum samples per dataset (for testing)')
    
    args = parser.parse_args()
    
    # Set data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif 'DATA_DIR' in os.environ:
        data_dir = os.environ['DATA_DIR']
    else:
        raise ValueError("Please set DATA_DIR environment variable or provide --data_dir argument")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup datasets
    print("Setting up datasets...")
    datasets = setup_datasets(data_dir)
    
    # Configure train datasets
    train_configs = []
    test_configs = []
    
    # MS-CXR
    if 'ms_cxr_train' in datasets:
        train_configs.append({
            "dataset": datasets['ms_cxr_train'],
            "name": "MS-CXR",
            "max_samples": args.max_samples_per_dataset
        })
    if 'ms_cxr_test' in datasets:
        test_configs.append({
            "dataset": datasets['ms_cxr_test'],
            "name": "MS-CXR",
            "max_samples": args.max_samples_per_dataset
        })
    
    # VinDr-CXR
    if 'vindr_train' in datasets:
        train_configs.append({
            "dataset": datasets['vindr_train'],
            "name": "VinDr-CXR",
            "max_samples": args.max_samples_per_dataset
        })
        train_configs.append({
            "dataset": datasets['vindr_single_train'],
            "name": "VinDr-CXR-Single",
            "max_samples": args.max_samples_per_dataset
        })
    if 'vindr_test' in datasets:
        test_configs.append({
            "dataset": datasets['vindr_test'],
            "name": "VinDr-CXR",
            "max_samples": args.max_samples_per_dataset
        })
        test_configs.append({
            "dataset": datasets['vindr_single_test'],
            "name": "VinDr-CXR-Single",
            "max_samples": args.max_samples_per_dataset
        })
    
    # PadChest
    if 'padchest_train' in datasets:
        train_configs.append({
            "dataset": datasets['padchest_train'],
            "name": "PadChest-GR",
            "max_samples": args.max_samples_per_dataset
        })
    if 'padchest_test' in datasets:
        test_configs.append({
            "dataset": datasets['padchest_test'],
            "name": "PadChest-GR",
            "max_samples": args.max_samples_per_dataset
        })
    
    # Create train dataset
    if train_configs:
        print("Creating training dataset...")
        train_samples = create_grounding_dataset(
            train_configs,
            split='train',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        # Save train dataset
        train_df = pd.DataFrame(train_samples)
        train_path = os.path.join(args.output_dir, 'train.parquet')
        train_df.to_parquet(train_path)
        print(f"Training dataset saved: {train_path} ({len(train_samples)} samples)")
    
    # Create test dataset
    if test_configs:
        print("Creating test dataset...")
        test_samples = create_grounding_dataset(
            test_configs,
            split='test',
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        # Save test dataset
        test_df = pd.DataFrame(test_samples)
        test_path = os.path.join(args.output_dir, 'test.parquet')
        test_df.to_parquet(test_path)
        print(f"Test dataset saved: {test_path} ({len(test_samples)} samples)")
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    if train_configs:
        print(f"Training samples: {len(train_samples)}")
        train_sources = train_df['data_source'].value_counts()
        print("Training data sources:")
        for source, count in train_sources.items():
            print(f"  {source}: {count}")
    
    if test_configs:
        print(f"Test samples: {len(test_samples)}")
        test_sources = test_df['data_source'].value_counts()
        print("Test data sources:")
        for source, count in test_sources.items():
            print(f"  {source}: {count}")

if __name__ == '__main__':
    main()