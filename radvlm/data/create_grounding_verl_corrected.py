import os
import argparse
import pandas as pd
import json
import random
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from radvlm.data.datasets import (
    MS_CXR, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset,
    PadChest_grounding, MIMIC_Dataset_MM
)
from radvlm.data.utils import custom_collate_fn
from radvlm import DATA_DIR

def create_grounding_prompt_for_verl(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Create a GRPO-compatible prompt that contains ONLY the question.
    For GRPO, we want the model to generate the answer and be rewarded based on IoU.
    """
    # Get the label/observation from the sample
    label = sample.get('label') or sample.get('observation', '')
    if not label:
        return None
    
    # Create question variations for grounding
    question_templates = [
        "Please provide the bounding box coordinates for the following description: {}",
        "Locate the following finding in the image: {}",
        "Where is the following observation located: {}",
        "Find the bounding box for: {}",
        "Provide coordinates for: {}",
        "Detect and locate: {}",
        "Identify the position of: {}"
    ]
    
    # Format the label properly
    if label and label[0].isupper() and not label.isupper():
        label = label.lower()
    
    # Choose a random question template
    question = random.choice(question_templates).format(label)
    
    # Return only the user's question - NO ANSWER for GRPO
    return [{"role": "user", "content": question}]

def create_verl_dataset_for_grpo(
    dataset_configs: List[Dict[str, Any]], 
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Create a veRL-compatible dataset for GRPO training.
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
                
                # Extract required fields
                gt_boxes = sample.get('boxes', [])
                if not gt_boxes:
                    continue
                
                # Create GRPO-compatible prompt (question only)
                verl_prompt = create_grounding_prompt_for_verl(sample)
                if not verl_prompt:
                    continue
                
                # Ensure boxes are properly formatted
                if not isinstance(gt_boxes, list) or not gt_boxes:
                    continue
                
                # Create veRL sample in correct format
                verl_sample = {
                    "data_source": dataset_name,  # Dataset-specific name
                    "prompt": verl_prompt,        # Only question, no answer
                    "ability": "phrase_grounding",
                    "reward_model": {
                        "style": "iou",
                        "ground_truth": gt_boxes,
                        "label": sample.get('label', sample.get('observation', ''))
                    },
                    "extra_info": {
                        "split": split,           # Required by veRL
                        "index": sample_count,    # Required by veRL
                        "image_path": sample.get('img_path', ''),
                        "dataset_source": dataset_name,
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

def setup_grounding_datasets(data_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    Setup all grounding datasets for the specified split.
    """
    # Dataset paths
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR', 'sentences_and_BBox_mscxr')
    dataset_path_vindr = os.path.join(data_dir, "VinDr-CXR")
    datasetpath_padchest = os.path.join(data_dir, 'PadChest')
    
    dataset_configs = []
    
    # MS-CXR (phrase grounding)
    if os.path.exists(sentencesBBoxpath):
        try:
            ms_cxr_dataset = MS_CXR(
                datasetpath=datasetpath_mimic,
                split=split,
                flag_img=False,
                flag_lab=True,
                only_frontal=True,
                flag_instr=True,
                sentencesBBoxpath=sentencesBBoxpath,
                seed=42
            )
            if len(ms_cxr_dataset) > 0:
                dataset_configs.append({
                    "dataset": ms_cxr_dataset,
                    "name": "MS-CXR",
                    "max_samples": len(ms_cxr_dataset)
                })
        except Exception as e:
            print(f"Warning: Could not load MS-CXR {split} split: {e}")
    
    # VinDr-CXR (abnormality grounding)
    if os.path.exists(dataset_path_vindr):
        try:
            vindr_dataset = VinDr_CXR_Dataset(
                datasetpath=dataset_path_vindr,
                split=split,
                flag_img=False
            )
            vindr_single_dataset = VinDr_CXR_Single_Label_Dataset(
                datasetpath=dataset_path_vindr,
                split=split,
                flag_img=False
            )
            
            if len(vindr_dataset) > 0:
                dataset_configs.append({
                    "dataset": vindr_dataset,
                    "name": "VinDr-CXR",
                    "max_samples": len(vindr_dataset)
                })
                
            if len(vindr_single_dataset) > 0:
                dataset_configs.append({
                    "dataset": vindr_single_dataset,
                    "name": "VinDr-CXR-Single",
                    "max_samples": len(vindr_single_dataset)
                })
        except Exception as e:
            print(f"Warning: Could not load VinDr-CXR {split} split: {e}")
    
    # PadChest (grounding) - uses 'valid' instead of 'test'
    if os.path.exists(datasetpath_padchest):
        padchest_split = 'valid' if split == 'test' else split
        try:
            padchest_dataset = PadChest_grounding(
                datasetpath=datasetpath_padchest,
                split=padchest_split,
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
            if len(padchest_dataset) > 0:
                dataset_configs.append({
                    "dataset": padchest_dataset,
                    "name": "PadChest-GR",
                    "max_samples": len(padchest_dataset)
                })
        except Exception as e:
            print(f"Warning: Could not load PadChest {split} split: {e}")
    
    return dataset_configs

def create_grounding_datasets_corrected(data_dir: str, output_dir: str, splits: List[str] = ['train', 'test']):
    """
    Create comprehensive grounding datasets with corrected veRL format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        print(f"\n=== Creating {split} dataset ===")
        
        # Setup datasets for this split
        dataset_configs = setup_grounding_datasets(data_dir, split)
        
        if not dataset_configs:
            print(f"❌ No datasets available for {split} split")
            continue
        
        # Create veRL dataset
        verl_samples = create_verl_dataset_for_grpo(
            dataset_configs,
            split=split,
            batch_size=32,
            num_workers=4,
            seed=42
        )
        
        if verl_samples:
            # Save to parquet
            df = pd.DataFrame(verl_samples)
            output_path = os.path.join(output_dir, f'{split}.parquet')
            df.to_parquet(output_path)
            print(f"✅ {split} dataset saved: {output_path} ({len(verl_samples)} samples)")
            
            # Print statistics
            sources = df['data_source'].value_counts()
            print(f"Dataset sources for {split}:")
            for source, count in sources.items():
                print(f"  {source}: {count} samples")
                
            # Verify format
            sample = verl_samples[0]
            print(f"\nSample format verification:")
            print(f"  data_source: {sample['data_source']}")
            print(f"  prompt: {sample['prompt']}")
            print(f"  ability: {sample['ability']}")
            print(f"  reward_model: {sample['reward_model']}")
            print(f"  extra_info keys: {list(sample['extra_info'].keys())}")
        else:
            print(f"❌ No samples generated for {split} split")

def main():
    parser = argparse.ArgumentParser(description="Create corrected grounding datasets for veRL GRPO training")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='./grounding_datasets_corrected', help='Output directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'], help='Dataset splits to create')
    
    args = parser.parse_args()
    
    # Set data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif 'DATA_DIR' in os.environ:
        data_dir = os.environ['DATA_DIR']
    else:
        raise ValueError("Please set DATA_DIR environment variable or provide --data_dir")
    
    create_grounding_datasets_corrected(data_dir, args.output_dir, args.splits)

if __name__ == '__main__':
    main()