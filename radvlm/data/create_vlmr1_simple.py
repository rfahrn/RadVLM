
"""
Simple, direct adaptation of create_llava_dataset.py to generate VLM-R1 compatible JSONL format.
This exactly matches the original RadVLM dataset logic and just changes the output format.
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
    """
    Create VLM-R1 compatible JSON cell from RadVLM sample.
    """
    sample_instr = sample["conversation"] if "conversation" in sample else sample["instr"]
    
    # Make image path relative to base_dir
    img_path = sample["img_path"]
    if isinstance(img_path, list):
        # Multi-image case
        image_paths = [os.path.relpath(path, base_dir) for path in img_path]
        image_count = len(image_paths)
    else:
        # Single image case
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
            # Already in conversation format
            conv_cell = instr.copy()
            if j == 0:
                # Add <image> tokens for first human message
                image_tokens = "<image>" * image_count
                conv_cell["value"] = f"{image_tokens}{conv_cell['value']}"
            json_cell["conversations"].append(conv_cell)
        elif "question" in instr and "answer" in instr:
            # Convert instruction format to conversation format
            conv_cell_human = {
                "from": "human",
                "value": instr["question"]
            }
            if j == 0:
                # Add <image> tokens for first human message
                image_tokens = "<image>" * image_count
                conv_cell_human["value"] = f"{image_tokens}{conv_cell_human['value']}"
            
            conv_cell_ai = {
                "from": "gpt",
                "value": instr["answer"]
            }
            json_cell["conversations"].append(conv_cell_human)
            json_cell["conversations"].append(conv_cell_ai)

    return json_cell


def generate_vlmr1_dataset_from_instruction_dataset(dataset_info, base_dir, batch_size=64, num_workers=8, seed=0):
    """
    Generate VLM-R1 JSONL dataset from RadVLM datasets.
    """
    vlmr1_samples = []

    for dataset_i, dataset_info_cell in enumerate(dataset_info):
        dataset = dataset_info_cell["dataset"]
        id_prefix = dataset_info_cell["id_prefix"]
        print(f"Processing {id_prefix}")
        np.random.seed(seed)
        random.seed(seed)
        
        num_samples = dataset_info_cell.get("num_samples", len(dataset))
        
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        sample_count = 0
        for batch in data_loader:
            for sample in batch:
                if sample_count >= num_samples:
                    break
                try:
                    json_cell = create_json_cell_vlmr1(
                        sample, 
                        dataset_info_cell["id_prefix"], 
                        len(vlmr1_samples), 
                        dataset,
                        base_dir
                    )
                    vlmr1_samples.append(json_cell)
                    sample_count += 1
                    
                    if sample_count % 1000 == 0:
                        print(f"  Processed {sample_count:,} samples...")
                        
                except Exception as e:
                    print(f"Warning: Skipping sample {sample_count}: {e}")
                    continue

            if sample_count >= num_samples:
                break
        
        print(f"Completed {id_prefix}: {sample_count:,} samples")
                
    return vlmr1_samples


def main():
    parser = argparse.ArgumentParser(description="Generate VLM-R1 JSONL training dataset from RadVLM (exactly like create_llava_dataset.py)")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", default="./vlmr1_datasets", help="Output directory")
    parser.add_argument("--output-file", default="all_train.jsonl", help="Output JSONL filename")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Generating VLM-R1 training dataset (same as create_llava_dataset.py)...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # EXACTLY THE SAME LOGIC AS create_llava_dataset.py
    print(f"=== GENERATING TRAINING DATA (exactly as create_llava_dataset.py) ===")
    
    # Copy the exact logic from create_llava_dataset.py
    datasetpath_mimic = os.path.join(args.data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(args.data_dir, 'MIMIC-CXR-JPG/filtered_reports')

    # Initialize datasets with error handling
    datasets_created = []
    
    # MIMIC-CXR reports
    try:
        print("MIMIC-CXR reports")
        mimic_dataset_filtered = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                                  split="train", 
                                                  flag_img=False, flag_lab=False, 
                                                  only_frontal=True, 
                                                  filtered_reports_dir=filtered_reports_dir, 
                                                  seed=0)
        print(f"Num samples = {len(mimic_dataset_filtered)}")
        datasets_created.append(("mimic_dataset_filtered", mimic_dataset_filtered))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping MIMIC-CXR reports: {e}")
        mimic_dataset_filtered = None

    # MIMIC-CXR classification
    try:
        print("MIMIC-CXR classif")
        mimic_dataset_labels = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                                split="train", 
                                                flag_img=False, flag_lab=True, 
                                                only_frontal=True, 
                                                filtered_reports_dir=None, 
                                                classif=True,
                                                seed=0)
        print(f"Num samples = {len(mimic_dataset_labels)}")
        datasets_created.append(("mimic_dataset_labels", mimic_dataset_labels))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping MIMIC-CXR classif: {e}")
        mimic_dataset_labels = None

    # CheXpert 
    try:
        print("CheXpert classif")
        dataset_path = os.path.join(args.data_dir, "CheXpert")   
        chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path,split="train", flag_img=False)
        print(f"Num samples = {len(chexpert_dataset)}")
        datasets_created.append(("chexpert_dataset", chexpert_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping CheXpert classif: {e}")
        chexpert_dataset = None

    # CheXpert-Plus reports
    try:
        print("CheXpert reports")
        datasetpath = os.path.join(args.data_dir, 'CheXpert')
        filtered_reports_dir_chex = os.path.join(datasetpath, 'filtered_reports')
        chexpertplus_dataset = CheXpertPlus_Dataset(datasetpath=datasetpath, split='train', flag_img=False, filtered_reports_dir=filtered_reports_dir_chex)
        print(f"Num samples = {len(chexpertplus_dataset)}")
        datasets_created.append(("chexpertplus_dataset", chexpertplus_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping CheXpert reports: {e}")
        chexpertplus_dataset = None

    # CHEST_IMA
    try:
        print("Chest-ima")
        datasetpath_chestima = os.path.join(args.data_dir, 'CHEST_IMA')
        chestima_dataset = Chest_ImaGenome_Dataset(
            datasetpath=datasetpath_mimic,
            datasetpath_chestima=datasetpath_chestima,
            split="train", 
            flag_img=False, 
            flag_instr=True, 
            flag_txt=False, 
            flag_lab=False,
            pick_one_region=True,
        )
        print(f"Num samples = {len(chestima_dataset)}")
        datasets_created.append(("chestima_dataset", chestima_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Chest-ima: {e}")
        chestima_dataset = None

    # VinDr-CXR
    try:
        print("VinDr-CXR")
        dataset_path = os.path.join(args.data_dir, "VinDr-CXR") 
        vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
        print(f"Num samples = {len(vin_dataset)}")
        datasets_created.append(("vin_dataset", vin_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping VinDr-CXR: {e}")
        vin_dataset = None

    try:
        print("VinDr-CXR mono")
        vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
        print(f"Num samples = {len(vin_dataset_mono)}")
        datasets_created.append(("vin_dataset_mono", vin_dataset_mono))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping VinDr-CXR mono: {e}")
        vin_dataset_mono = None

    # Phrase grounding MS-CXR
    try:
        print("Phrase grounding MS-CXR")
        sentencesBBoxpath = os.path.join(args.data_dir, 'MS-CXR','sentences_and_BBox_mscxr')

        dataset_train = MS_CXR(
            datasetpath = datasetpath_mimic,
            split="train", flag_img=False, 
            flag_lab=True, only_frontal=True, 
            flag_instr=True, 
            sentencesBBoxpath=sentencesBBoxpath,
            seed=0)

        dataset_valid = MS_CXR(
            datasetpath = datasetpath_mimic,
            split="valid", flag_img=False, 
            flag_lab=True, only_frontal=True, 
            flag_instr=True, 
            sentencesBBoxpath=sentencesBBoxpath,
            seed=0)

        prhase_grounding_mscxr_dataset = ConcatDataset([dataset_train, dataset_valid])
        print(f"Num samples = {len(prhase_grounding_mscxr_dataset)}")
        datasets_created.append(("prhase_grounding_mscxr_dataset", prhase_grounding_mscxr_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Phrase grounding MS-CXR: {e}")
        prhase_grounding_mscxr_dataset = None

    # Phrase grounding PadChest
    try:
        print("Phrase grounding PadChest")
        datasetpath_pad = os.path.join(args.data_dir, 'PadChest')
        dataset_train_pad = PadChest_grounding(
            datasetpath=datasetpath_pad,
            split='train', 
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )

        dataset_valid_pad = PadChest_grounding(
            datasetpath=datasetpath_pad,
            split='valid', 
            flag_instr=True,
            flag_img=False,
            flag_txt=False
        )

        prhase_grounding_padchest_dataset = ConcatDataset([dataset_train_pad, dataset_valid_pad])
        print(f"Num samples = {len(prhase_grounding_padchest_dataset)}")
        datasets_created.append(("prhase_grounding_padchest_dataset", prhase_grounding_padchest_dataset))
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Phrase grounding PadChest: {e}")
        prhase_grounding_padchest_dataset = None

    # CONVERSATIONS
    try:
        print("Conversations standard")
        conversation_dir= os.path.join(datasetpath_mimic, 'conversations/train/standard')
        conv_dataset_standard = None
        if os.path.exists(conversation_dir):
            conv_dataset_standard = MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split="train", 
                flag_img=False, 
                flag_instr=False, 
                flag_txt=False, 
                flag_lab=False, 
                filtered_reports_dir=filtered_reports_dir,
                conversation_dir=conversation_dir
                )
            print(f"Num samples = {len(conv_dataset_standard)}")
            datasets_created.append(("conv_dataset_standard", conv_dataset_standard))
        else:
            print("Conversations standard directory not found - skipping")
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Conversations standard: {e}")
        conv_dataset_standard = None

    try:
        print("Conversations grounded")
        conversation_dir_grounded =  os.path.join(datasetpath_mimic, 'conversations/train/grounding')
        conv_dataset_grounded = None
        if os.path.exists(conversation_dir_grounded):
            conv_dataset_grounded = MIMIC_Dataset_MM(
                datasetpath = datasetpath_mimic,
                split="train", flag_img=False, 
                flag_lab=False, only_frontal=True, 
                flag_instr=False, 
                filtered_reports_dir=filtered_reports_dir,
                sentencesBBoxpath = sentencesBBoxpath,
                conversation_dir = conversation_dir_grounded,
                classif=False,
                seed=0)
            print(f"Num samples = {len(conv_dataset_grounded)}")
            datasets_created.append(("conv_dataset_grounded", conv_dataset_grounded))
        else:
            print("Conversations grounded directory not found - skipping")
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Conversations grounded: {e}")
        conv_dataset_grounded = None

    try:
        print("Conversations grounded padchest")
        datasetpath_pad = os.path.join(args.data_dir, 'PadChest')
        conversation_dir_pad = os.path.join(datasetpath_pad, 'conversations/train/grounding')
        conv_dataset_grounded_padchest = None
        if os.path.exists(conversation_dir_pad):
            dataset_train_conv = PadChest_grounding_per_image(
                datasetpath=datasetpath_pad,
                split='train',
                flag_instr=False,
                flag_img=False,
                conversation_dir=conversation_dir_pad
            )
            dataset_valid_conv = PadChest_grounding_per_image(
                datasetpath=datasetpath_pad,
                split='valid',
                flag_instr=False,
                flag_img=False,
                conversation_dir=conversation_dir_pad
            )
            conv_dataset_grounded_padchest = ConcatDataset([dataset_train_conv, dataset_valid_conv])
            print(f"Num samples = {len(conv_dataset_grounded_padchest)}")
            datasets_created.append(("conv_dataset_grounded_padchest", conv_dataset_grounded_padchest))
        else:
            print("PadChest conversations grounded directory not found - skipping")
    except (PermissionError, FileNotFoundError, Exception) as e:
        print(f"❌ Skipping Conversations grounded padchest: {e}")
        conv_dataset_grounded_padchest = None

    # Create dataset_info list (only for successfully created datasets)
    dataset_info = []
    
    if vin_dataset:
        dataset_info.extend([
            {"dataset":vin_dataset, "id_prefix":"vindr-cxr-train1"},
            {"dataset":vin_dataset, "id_prefix":"vindr-cxr-train2"},
        ])
        
    if vin_dataset_mono:
        dataset_info.extend([
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train1"},
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train2"},
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train3"},
        ])
        
    if prhase_grounding_mscxr_dataset:
        dataset_info.extend([
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train1"},
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train2"},
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train3"},
        ])
        
    if prhase_grounding_padchest_dataset:
        dataset_info.extend([
            {"dataset":prhase_grounding_padchest_dataset, "id_prefix":"padchest-train1"},
            {"dataset":prhase_grounding_padchest_dataset, "id_prefix":"padchest-train2"},
        ])
        
    if mimic_dataset_filtered:
        dataset_info.append({"dataset":mimic_dataset_filtered, "id_prefix":"mimic-train"})
        
    if chexpertplus_dataset:
        dataset_info.append({"dataset":chexpertplus_dataset, "id_prefix":"chexpertplus-train"})
        
    if chestima_dataset:
        dataset_info.append({"dataset":chestima_dataset, "id_prefix":"chestima-train", "num_samples":80000})
        
    if mimic_dataset_labels:
        dataset_info.append({"dataset":mimic_dataset_labels, "id_prefix":"mimic-labels-train"})
        
    if chexpert_dataset:
        dataset_info.append({"dataset":chexpert_dataset, "id_prefix":"chexpert-train"})

    # Add conversation datasets if they exist
    if conv_dataset_standard:
        dataset_info.append({"dataset":conv_dataset_standard, "id_prefix":"conv-train"})
    
    if conv_dataset_grounded:
        dataset_info.extend([
            {"dataset":conv_dataset_grounded, "id_prefix":"conv-grounded-train1"},
            {"dataset":conv_dataset_grounded, "id_prefix":"conv-grounded-train2"},
            {"dataset":conv_dataset_grounded, "id_prefix":"conv-grounded-train3"},
            {"dataset":conv_dataset_grounded, "id_prefix":"conv-grounded-train4"},
        ])
    
    if conv_dataset_grounded_padchest:
        dataset_info.extend([
            {"dataset":conv_dataset_grounded_padchest, "id_prefix":"conv-grounded-padchest-train1"},
            {"dataset":conv_dataset_grounded_padchest, "id_prefix":"conv-grounded-padchest-train2"},
            {"dataset":conv_dataset_grounded_padchest, "id_prefix":"conv-grounded-padchest-train3"},
            {"dataset":conv_dataset_grounded_padchest, "id_prefix":"conv-grounded-padchest-train4"},
        ])

    if not dataset_info:
        print("❌ No datasets were successfully loaded! Check data directory and permissions.")
        return

    # Generate VLM-R1 training dataset
    print(f"\n🚀 Generating VLM-R1 training dataset from {len(dataset_info)} configurations...")
    train_vlmr1_dataset = generate_vlmr1_dataset_from_instruction_dataset(
        dataset_info, 
        args.data_dir,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        seed=args.seed
    )

    # Save training data
    train_output_path = os.path.join(args.output_dir, args.output_file)
    with open(train_output_path, "w", encoding='utf-8') as f:
        for sample in train_vlmr1_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ Training dataset saved!")
    print(f"📁 File: {train_output_path}")
    print(f"📊 Training samples: {len(train_vlmr1_dataset):,}")
    print(f"📊 Successfully loaded datasets: {len(datasets_created)}")
    
    print(f"\n🚀 Usage with VLM-R1:")
    print(f"python -m open_r1.grpo_jsonl \\")
    print(f"    --data_file_paths {train_output_path} \\")
    print(f"    --image_folders {args.data_dir}/ \\")
    print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")


if __name__ == "__main__":
    main()
