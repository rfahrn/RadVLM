#!/usr/bin/env python
"""
Generate individual task-specific VLM-R1 JSONL files for each RadVLM task.
Creates separate train.jsonl and test.jsonl files for each task so you can use them individually with VLM-R1.

Example usage with VLM-R1:
--data_file_paths report_generation_mimic_train.jsonl:classification_chexpert_train.jsonl:grounding_mscxr_train.jsonl
--image_folders /path/to/data/:/path/to/data/:/path/to/data/
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


def generate_vlmr1_samples(dataset, id_prefix, base_dir, num_samples=None, batch_size=64, num_workers=8, seed=0):
    """Generate VLM-R1 samples from a single dataset."""
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


def create_task_datasets(data_dir, output_dir, batch_size=64, num_workers=8, seed=0):
    """Create all individual task datasets."""
    
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')
    
    task_configs = []
    
    # 1. REPORT GENERATION - MIMIC-CXR
    try:
        print("📝 Report Generation - MIMIC-CXR")
        mimic_reports_train = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="train", 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        )
        print(f"  Train samples: {len(mimic_reports_train):,}")
        
        mimic_reports_test = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="valid", 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        )
        print(f"  Test samples: {len(mimic_reports_test):,}")
        
        task_configs.append({
            "task_name": "report_generation_mimic",
            "train_dataset": mimic_reports_train,
            "test_dataset": mimic_reports_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Report Generation MIMIC: {e}")

    # 2. REPORT GENERATION - CheXpert-Plus  
    try:
        print("📝 Report Generation - CheXpert-Plus")
        datasetpath_chex = os.path.join(data_dir, 'CheXpert')
        filtered_reports_dir_chex = os.path.join(datasetpath_chex, 'filtered_reports')
        
        chexpert_plus_train = CheXpertPlus_Dataset(
            datasetpath=datasetpath_chex, 
            split='train', 
            flag_img=False, 
            filtered_reports_dir=filtered_reports_dir_chex
        )
        print(f"  Train samples: {len(chexpert_plus_train):,}")
        
        task_configs.append({
            "task_name": "report_generation_chexpert_plus",
            "train_dataset": chexpert_plus_train,
            "test_dataset": None  # CheXpert-Plus only has training data
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Report Generation CheXpert-Plus: {e}")

    # 3. ABNORMALITY CLASSIFICATION - MIMIC-CXR
    try:
        print("🏷️ Abnormality Classification - MIMIC-CXR")
        mimic_classif_train = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="train", 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        )
        print(f"  Train samples: {len(mimic_classif_train):,}")
        
        mimic_classif_test = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split="valid", 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        )
        print(f"  Test samples: {len(mimic_classif_test):,}")
        
        task_configs.append({
            "task_name": "abnormality_classification_mimic",
            "train_dataset": mimic_classif_train,
            "test_dataset": mimic_classif_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Classification MIMIC: {e}")

    # 4. ABNORMALITY CLASSIFICATION - CheXpert
    try:
        print("🏷️ Abnormality Classification - CheXpert")
        dataset_path = os.path.join(data_dir, "CheXpert")
        
        chexpert_train = CheXpert_Dataset_MM(datasetpath=dataset_path, split="train", flag_img=False)
        print(f"  Train samples: {len(chexpert_train):,}")
        
        chexpert_test = CheXpert_Dataset_MM(datasetpath=dataset_path, split="valid", flag_img=False)
        print(f"  Test samples: {len(chexpert_test):,}")
        
        task_configs.append({
            "task_name": "abnormality_classification_chexpert",
            "train_dataset": chexpert_train,
            "test_dataset": chexpert_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Classification CheXpert: {e}")

    # 5. ANATOMICAL GROUNDING - Chest ImaGenome
    try:
        print("🎯 Anatomical Grounding - Chest ImaGenome")
        datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
        
        chestima_train = Chest_ImaGenome_Dataset(
            datasetpath=datasetpath_mimic,
            datasetpath_chestima=datasetpath_chestima,
            split="train", 
            flag_img=False, 
            flag_instr=True, 
            flag_txt=False, 
            flag_lab=False,
            pick_one_region=True,
        )
        print(f"  Train samples: {len(chestima_train):,}")
        
        # Chest ImaGenome test split
        chestima_test = Chest_ImaGenome_Dataset(
            datasetpath=datasetpath_mimic,
            datasetpath_chestima=datasetpath_chestima,
            split="test", 
            flag_img=False, 
            flag_instr=True, 
            flag_txt=False, 
            flag_lab=False,
            pick_one_region=True,
        )
        print(f"  Test samples: {len(chestima_test):,}")
        
        task_configs.append({
            "task_name": "anatomical_grounding_chest_imagenome",
            "train_dataset": chestima_train,
            "test_dataset": chestima_test,
            "num_samples": 80000  # Limit training samples as in original
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Anatomical Grounding: {e}")

    # 6. ABNORMALITY GROUNDING - VinDr-CXR
    try:
        print("🎯 Abnormality Grounding - VinDr-CXR")
        dataset_path = os.path.join(data_dir, "VinDr-CXR")
        
        vindr_train = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img=False)
        print(f"  Train samples: {len(vindr_train):,}")
        
        vindr_test = VinDr_CXR_Dataset(datasetpath=dataset_path, split="test", flag_img=False)
        print(f"  Test samples: {len(vindr_test):,}")
        
        task_configs.append({
            "task_name": "abnormality_grounding_vindr_cxr",
            "train_dataset": vindr_train,
            "test_dataset": vindr_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Abnormality Grounding VinDr: {e}")

    # 7. ABNORMALITY DETECTION - VinDr-CXR (Single Label)
    try:
        print("🔍 Abnormality Detection - VinDr-CXR")
        
        vindr_mono_train = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="train", flag_img=False)
        print(f"  Train samples: {len(vindr_mono_train):,}")
        
        vindr_mono_test = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="test", flag_img=False)
        print(f"  Test samples: {len(vindr_mono_test):,}")
        
        task_configs.append({
            "task_name": "abnormality_detection_vindr_cxr", 
            "train_dataset": vindr_mono_train,
            "test_dataset": vindr_mono_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Abnormality Detection VinDr: {e}")

    # 8. PHRASE GROUNDING - MS-CXR
    try:
        print("📍 Phrase Grounding - MS-CXR")
        
        mscxr_train = MS_CXR(
            datasetpath=datasetpath_mimic,
            split="train", flag_img=False, 
            flag_lab=True, only_frontal=True, 
            flag_instr=True, 
            sentencesBBoxpath=sentencesBBoxpath,
            seed=0
        )
        
        mscxr_test = MS_CXR(
            datasetpath=datasetpath_mimic,
            split="valid", flag_img=False, 
            flag_lab=True, only_frontal=True, 
            flag_instr=True, 
            sentencesBBoxpath=sentencesBBoxpath,
            seed=0
        )
        
        print(f"  Train samples: {len(mscxr_train):,}")
        print(f"  Test samples: {len(mscxr_test):,}")
        
        task_configs.append({
            "task_name": "phrase_grounding_mscxr",
            "train_dataset": mscxr_train,
            "test_dataset": mscxr_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Phrase Grounding MS-CXR: {e}")

    # 9. PHRASE GROUNDING - PadChest
    try:
        print("📍 Phrase Grounding - PadChest")
        datasetpath_pad = os.path.join(data_dir, 'PadChest')
        
        padchest_train = PadChest_grounding(
            datasetpath=datasetpath_pad,
            split='train', 
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )
        
        padchest_test = PadChest_grounding(
            datasetpath=datasetpath_pad,
            split='valid', 
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )
        
        print(f"  Train samples: {len(padchest_train):,}")
        print(f"  Test samples: {len(padchest_test):,}")
        
        task_configs.append({
            "task_name": "phrase_grounding_padchest",
            "train_dataset": padchest_train,
            "test_dataset": padchest_test
        })
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Phrase Grounding PadChest: {e}")

    # 10. CONVERSATIONS - MIMIC-CXR Standard
    try:
        print("💬 Conversations - MIMIC-CXR Standard")
        conversation_dir = os.path.join(datasetpath_mimic, 'conversations/train/standard')
        conversation_dir_test = os.path.join(datasetpath_mimic, 'conversations/valid/standard')
        
        if os.path.exists(conversation_dir):
            conv_train = MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="train", 
                flag_img=False, 
                flag_instr=False, 
                flag_txt=False, 
                flag_lab=False, 
                filtered_reports_dir=filtered_reports_dir,
                conversation_dir=conversation_dir
            )
            print(f"  Train samples: {len(conv_train):,}")
            
            conv_test = None
            if os.path.exists(conversation_dir_test):
                conv_test = MIMIC_Dataset_MM(
                    datasetpath=datasetpath_mimic,
                    split="valid", 
                    flag_img=False, 
                    flag_instr=False, 
                    flag_txt=False, 
                    flag_lab=False, 
                    filtered_reports_dir=filtered_reports_dir,
                    conversation_dir=conversation_dir_test
                )
                print(f"  Test samples: {len(conv_test):,}")
            
            task_configs.append({
                "task_name": "conversations_mimic_standard",
                "train_dataset": conv_train,
                "test_dataset": conv_test
            })
        else:
            print("  ❌ Conversation directory not found - skipping")
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Conversations Standard: {e}")

    # 11. CONVERSATIONS - MIMIC-CXR Grounded
    try:
        print("💬 Conversations - MIMIC-CXR Grounded")
        conversation_dir_grounded = os.path.join(datasetpath_mimic, 'conversations/train/grounding')
        conversation_dir_grounded_test = os.path.join(datasetpath_mimic, 'conversations/valid/grounding')
        
        if os.path.exists(conversation_dir_grounded):
            conv_grounded_train = MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="train", flag_img=False, 
                flag_lab=False, only_frontal=True, 
                flag_instr=False, 
                filtered_reports_dir=filtered_reports_dir,
                sentencesBBoxpath=sentencesBBoxpath,
                conversation_dir=conversation_dir_grounded,
                classif=False,
                seed=0
            )
            print(f"  Train samples: {len(conv_grounded_train):,}")
            
            conv_grounded_test = None
            if os.path.exists(conversation_dir_grounded_test):
                conv_grounded_test = MIMIC_Dataset_MM(
                    datasetpath=datasetpath_mimic,
                    split="valid", flag_img=False, 
                    flag_lab=False, only_frontal=True, 
                    flag_instr=False, 
                    filtered_reports_dir=filtered_reports_dir,
                    sentencesBBoxpath=sentencesBBoxpath,
                    conversation_dir=conversation_dir_grounded_test,
                    classif=False,
                    seed=0
                )
                print(f"  Test samples: {len(conv_grounded_test):,}")
            
            task_configs.append({
                "task_name": "conversations_mimic_grounded",
                "train_dataset": conv_grounded_train,
                "test_dataset": conv_grounded_test
            })
        else:
            print("  ❌ Grounded conversation directory not found - skipping")
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Conversations Grounded: {e}")

    # Now generate JSONL files for each task
    print(f"\n🚀 Generating individual JSONL files for {len(task_configs)} tasks...")
    
    for config in task_configs:
        task_name = config["task_name"]
        train_dataset = config["train_dataset"]
        test_dataset = config.get("test_dataset")
        num_samples = config.get("num_samples")
        
        print(f"\n📁 Processing: {task_name}")
        
        # Generate training data
        train_samples = generate_vlmr1_samples(
            train_dataset, 
            f"{task_name}_train",
            data_dir,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed
        )
        
        train_output = os.path.join(output_dir, f"{task_name}_train.jsonl")
        save_jsonl(train_samples, train_output)
        
        # Generate test data if available
        if test_dataset:
            test_samples = generate_vlmr1_samples(
                test_dataset,
                f"{task_name}_test", 
                data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed
            )
            
            test_output = os.path.join(output_dir, f"{task_name}_test.jsonl")
            save_jsonl(test_samples, test_output)
        else:
            print(f"  ℹ️  No test dataset available for {task_name}")

    return task_configs


def main():
    parser = argparse.ArgumentParser(description="Generate individual task-specific VLM-R1 JSONL files")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", default="./vlmr1_individual_tasks", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"🎯 Generating individual task-specific VLM-R1 JSONL files...")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Create task datasets
    task_configs = create_task_datasets(
        args.data_dir, 
        args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"\n✅ Generated individual JSONL files for {len(task_configs)} tasks!")
    print(f"📁 Files saved to: {args.output_dir}")
    
    # Show usage examples
    print(f"\n🚀 Usage examples with VLM-R1:")
    print(f"\n# Single task training:")
    print(f"python -m open_r1.grpo_jsonl \\")
    print(f"    --data_file_paths {args.output_dir}/report_generation_mimic_train.jsonl \\")
    print(f"    --image_folders {args.data_dir}/ \\")
    print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
    
    print(f"\n# Multi-task training:")
    print(f"python -m open_r1.grpo_jsonl \\")
    print(f"    --data_file_paths {args.output_dir}/report_generation_mimic_train.jsonl:{args.output_dir}/abnormality_classification_chexpert_train.jsonl:{args.output_dir}/phrase_grounding_mscxr_train.jsonl \\")
    print(f"    --image_folders {args.data_dir}/:{args.data_dir}/:{args.data_dir}/ \\")
    print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
    
    print(f"\n📋 Available task files:")
    for config in task_configs:
        task_name = config["task_name"]
        print(f"  • {task_name}_train.jsonl", end="")
        if config.get("test_dataset"):
            print(f" + {task_name}_test.jsonl")
        else:
            print("")


if __name__ == "__main__":
    main()