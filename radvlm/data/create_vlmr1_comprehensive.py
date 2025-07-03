
"""
Comprehensive script to generate VLM-R1 compatible JSONL datasets from RadVLM.
This script handles all dataset types supported by RadVLM and converts them to the format
expected by VLM-R1 training.

VLM-R1 JSONL format:
{
  "id": 1,
  "image": "relative/path/to/image.png",
  "conversations": [
    {"from": "human", "value": "<image>Question here?"},
    {"from": "gpt", "value": "Answer here"}
  ]
}

For multi-image:
{
  "id": 1,
  "image": ["path1.png", "path2.png"],
  "conversations": [
    {"from": "human", "value": "<image><image>Question with multiple images?"},
    {"from": "gpt", "value": "Answer here"}
  ]
}
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

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


def convert_conversation_to_vlmr1(conversations, image_count=1):
    """
    Convert LLaVA conversation format to VLM-R1 format.
    
    Args:
        conversations: List of conversation turns from LLaVA format
        image_count: Number of images (for multi-image support)
    
    Returns:
        List of conversation turns in VLM-R1 format
    """
    vlmr1_conversations = []
    
    for i, conv in enumerate(conversations):
        if conv.get("from") == "human":
            # Add <image> tokens for the first human message
            if i == 0:
                image_tokens = "<image>" * image_count
                value = f"{image_tokens}{conv['value']}"
            else:
                value = conv["value"]
            
            vlmr1_conversations.append({
                "from": "human",
                "value": value
            })
        elif conv.get("from") == "gpt":
            vlmr1_conversations.append({
                "from": "gpt", 
                "value": conv["value"]
            })
    
    return vlmr1_conversations


def convert_instruction_to_vlmr1(instruction, image_count=1):
    """
    Convert instruction format to VLM-R1 conversation format.
    
    Args:
        instruction: Instruction dict with 'question' and 'answer' keys
        image_count: Number of images
    
    Returns:
        List of conversation turns in VLM-R1 format
    """
    image_tokens = "<image>" * image_count
    
    return [
        {
            "from": "human",
            "value": f"{image_tokens}{instruction['question']}"
        },
        {
            "from": "gpt",
            "value": instruction["answer"]
        }
    ]


def make_image_path_relative(img_path, base_dir):
    """
    Convert absolute image path to relative path from base_dir.
    
    Args:
        img_path: Absolute path to image
        base_dir: Base directory to make path relative to
    
    Returns:
        Relative path string
    """
    try:
        return os.path.relpath(img_path, base_dir)
    except ValueError:
        # If paths are on different drives (Windows), return original path
        return img_path


def create_vlmr1_sample(sample, sample_id, base_dir, id_prefix=""):
    """
    Create a VLM-R1 compatible sample from RadVLM sample.
    
    Args:
        sample: Sample dict from RadVLM dataset
        sample_id: Unique sample ID
        base_dir: Base directory for relative paths
        id_prefix: Prefix for sample ID
    
    Returns:
        Dict in VLM-R1 format
    """
    # Handle image path(s)
    if "img_path" in sample:
        if isinstance(sample["img_path"], list):
            # Multi-image case
            image_paths = [make_image_path_relative(path, base_dir) for path in sample["img_path"]]
            image_count = len(image_paths)
        else:
            # Single image case
            image_paths = make_image_path_relative(sample["img_path"], base_dir)
            image_count = 1
    else:
        raise ValueError("Sample missing 'img_path' field")
    
    # Handle conversations
    if "conversation" in sample and sample["conversation"]:
        # Use existing conversation format
        conversations = convert_conversation_to_vlmr1(sample["conversation"], image_count)
    elif "instr" in sample:
        # Convert instruction format
        if isinstance(sample["instr"], list):
            # Multiple instructions
            conversations = []
            for i, instr in enumerate(sample["instr"]):
                if isinstance(instr, dict) and "question" in instr and "answer" in instr:
                    img_count = image_count if i == 0 else 0  # Only add image tokens to first instruction
                    conv = convert_instruction_to_vlmr1(instr, img_count)
                    conversations.extend(conv)
        elif isinstance(sample["instr"], dict):
            conversations = convert_instruction_to_vlmr1(sample["instr"], image_count)
        else:
            raise ValueError(f"Unsupported instruction format: {type(sample['instr'])}")
    else:
        raise ValueError("Sample missing both 'conversation' and 'instr' fields")
    
    # Create VLM-R1 sample
    vlmr1_sample = {
        "id": f"{id_prefix}_{sample_id}" if id_prefix else str(sample_id),
        "image": image_paths,
        "conversations": conversations
    }
    
    return vlmr1_sample


def get_dataset_configs(data_dir):
    """
    Get all dataset configurations matching the original create_llava_dataset.py.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        List of dataset configuration dicts
    """
    # MIMIC-CXR paths
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    
    datasets = []
    
    # MIMIC-CXR reports
    datasets.append({
        "name": "mimic_reports",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="train", 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        ),
        "id_prefix": "mimic-train"
    })
    
    # MIMIC-CXR classification
    datasets.append({
        "name": "mimic_labels",
        "dataset": MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="train", 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        ),
        "id_prefix": "mimic-labels-train"
    })
    
    # CheXpert classification
    dataset_path = os.path.join(data_dir, "CheXpert")   
    datasets.append({
        "name": "chexpert",
        "dataset": CheXpert_Dataset_MM(
            datasetpath=dataset_path,
            split="train", 
            flag_img=False
        ),
        "id_prefix": "chexpert-train"
    })
    
    # CheXpert-Plus reports
    datasetpath = os.path.join(data_dir, 'CheXpert')
    filtered_reports_dir = os.path.join(datasetpath, 'filtered_reports')
    datasets.append({
        "name": "chexpertplus",
        "dataset": CheXpertPlus_Dataset(
            datasetpath=datasetpath, 
            split='train', 
            flag_img=False, 
            filtered_reports_dir=filtered_reports_dir
        ),
        "id_prefix": "chexpertplus-train"
    })
    
    # Chest-ImaGenome
    datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
    datasets.append({
        "name": "chestima",
        "dataset": Chest_ImaGenome_Dataset(
            datasetpath=datasetpath_mimic,
            datasetpath_chestima=datasetpath_chestima,
            split="train", 
            flag_img=False, 
            flag_instr=True, 
            flag_txt=False, 
            flag_lab=False,
            pick_one_region=True,
        ),
        "id_prefix": "chestima-train",
        "num_samples": 80000
    })
    
    # VinDr-CXR
    dataset_path = os.path.join(data_dir, "VinDr-CXR") 
    datasets.append({
        "name": "vindr_cxr",
        "dataset": VinDr_CXR_Dataset(
            datasetpath=dataset_path, 
            split="train", 
            flag_img=False
        ),
        "id_prefix": "vindr-cxr-train1"
    })
    
    datasets.append({
        "name": "vindr_cxr_mono",
        "dataset": VinDr_CXR_Single_Label_Dataset(
            datasetpath=dataset_path, 
            split="train", 
            flag_img=False
        ),
        "id_prefix": "vindr-cxr-mono-train1"
    })
    
    # MS-CXR phrase grounding
    dataset_train = MS_CXR(
        datasetpath=datasetpath_mimic,
        split="train", flag_img=False, 
        flag_lab=True, only_frontal=True, 
        flag_instr=True, 
        sentencesBBoxpath=sentencesBBoxpath,
        seed=0
    )
    
    dataset_valid = MS_CXR(
        datasetpath=datasetpath_mimic,
        split="valid", flag_img=False, 
        flag_lab=True, only_frontal=True, 
        flag_instr=True, 
        sentencesBBoxpath=sentencesBBoxpath,
        seed=0
    )
    
    prhase_grounding_mscxr_dataset = ConcatDataset([dataset_train, dataset_valid])
    datasets.append({
        "name": "mscxr_grounding",
        "dataset": prhase_grounding_mscxr_dataset,
        "id_prefix": "mscxr-train1"
    })
    
    # PadChest phrase grounding
    datasetpath_pad = os.path.join(data_dir, 'PadChest')
    dataset_train_pad = PadChest_grounding(
        datasetpath=datasetpath_pad,
        split="train", 
        flag_instr=True,
        flag_img=False,
        flag_txt=False
    )
    
    dataset_valid_pad = PadChest_grounding(
        datasetpath=datasetpath_pad,
        split="valid", 
        flag_instr=True,
        flag_img=False,
        flag_txt=False
    )
    
    prhase_grounding_padchest_dataset = ConcatDataset([dataset_train_pad, dataset_valid_pad])
    datasets.append({
        "name": "padchest_grounding",
        "dataset": prhase_grounding_padchest_dataset,
        "id_prefix": "padchest-train1"
    })
    
    # Conversations standard
    conversation_dir = os.path.join(datasetpath_mimic, 'conversations/train/standard')
    if os.path.exists(conversation_dir):
        datasets.append({
            "name": "conversations_standard",
            "dataset": MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="train", 
                flag_img=False, 
                flag_instr=False, 
                flag_txt=False, 
                flag_lab=False, 
                filtered_reports_dir=filtered_reports_dir,
                conversation_dir=conversation_dir
            ),
            "id_prefix": "conv-train"
        })
    
    # Conversations grounded
    conversation_dir_grounded = os.path.join(datasetpath_mimic, 'conversations/train/grounding')
    if os.path.exists(conversation_dir_grounded):
        datasets.append({
            "name": "conversations_grounded",
            "dataset": MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="train", flag_img=False, 
                flag_lab=False, only_frontal=True, 
                flag_instr=False, 
                filtered_reports_dir=filtered_reports_dir,
                sentencesBBoxpath=sentencesBBoxpath,
                conversation_dir=conversation_dir_grounded,
                classif=False,
                seed=0
            ),
            "id_prefix": "conv-grounded-train1"
        })
    
    # PadChest conversations grounded
    conversation_dir_pad = os.path.join(datasetpath_pad, 'conversations/train/grounding')
    if os.path.exists(conversation_dir_pad):
        dataset_train_conv = PadChest_grounding_per_image(
            datasetpath=datasetpath_pad,
            split="train",
            flag_instr=False,
            flag_img=False,
            conversation_dir=conversation_dir_pad
        )
        
        dataset_valid_conv = PadChest_grounding_per_image(
            datasetpath=datasetpath_pad,
            split="valid",
            flag_instr=False,
            flag_img=False,
            conversation_dir=conversation_dir_pad
        )
        
        conv_dataset_grounded_padchest = ConcatDataset([dataset_train_conv, dataset_valid_conv])
        datasets.append({
            "name": "padchest_conversations_grounded",
            "dataset": conv_dataset_grounded_padchest,
            "id_prefix": "conv-grounded-padchest-train1"
        })
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM-R1 compatible JSONL datasets from all RadVLM datasets"
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
        help="Create a single combined JSONL file with all datasets"
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
    dataset_configs = get_dataset_configs(args.data_dir)
    
    # Filter datasets if specified
    if args.datasets:
        dataset_configs = [cfg for cfg in dataset_configs if cfg["name"] in args.datasets]
    
    print(f"Processing {len(dataset_configs)} datasets...")
    
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
            output_file = os.path.join(args.output_dir, f"{config['name']}_train.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in dataset_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"Saved {len(dataset_samples)} samples to {output_file}")
    
    # Save combined file if requested
    if args.combine_all:
        output_file = os.path.join(args.output_dir, "all_train.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(all_samples)} total samples to {output_file}")
    
    print(f"\n✅ Processing complete! Generated {len(all_samples)} total samples.")
    print(f"Files saved to: {args.output_dir}")
    print(f"\nUsage with VLM-R1:")
    print(f"--data_file_paths {args.output_dir}/all_train.jsonl")
    print(f"--image_folders {args.data_dir}/")


if __name__ == "__main__":
    main()
