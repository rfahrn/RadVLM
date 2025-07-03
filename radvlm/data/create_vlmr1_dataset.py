#!/usr/bin/env python
"""
Simple adaptation of create_llava_dataset.py to generate VLM-R1 compatible JSONL format.
This reuses all the existing RadVLM dataset logic and just changes the output format.
"""

import torch
import numpy as np
from radvlm.data.datasets import MIMIC_Dataset_MM, CheXpert_Dataset_MM, Chest_ImaGenome_Dataset, MS_CXR, CheXpertPlus_Dataset, PadChest_grounding, PadChest_grounding_per_image, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset
from radvlm.data.utils import *
import json
import os
import argparse
from torch.utils.data import ConcatDataset
from radvlm import DATA_DIR


def create_json_cell_vlmr1(sample, id_prefix, sample_idx, dataset, base_dir):
    """
    Create VLM-R1 compatible JSON cell from RadVLM sample.
    Similar to create_json_cell_llava but outputs VLM-R1 format.
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
        # Case when sample_instr is a dict with question and answer
        sample_instr = [sample_instr]  # Convert to a list for uniform processing
    
    for j, instr in enumerate(sample_instr):
        if "from" in instr and "value" in instr:
            # Case when it's already in the correct format
            conv_cell = instr.copy()
            if j == 0:
                # Add <image> tokens for first human message
                image_tokens = "<image>" * image_count
                conv_cell["value"] = f"{image_tokens}{conv_cell['value']}"
            json_cell["conversations"].append(conv_cell)
        elif "question" in instr and "answer" in instr:
            # Case when it's a dict with question and answer keys
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
        else:
            continue

    return json_cell


def create_dataset_configs(data_dir, split="train"):
    """
    Create dataset configurations for a specific split.
    Maps 'test' to 'valid' for datasets that don't have a test split.
    """
    # Map test to valid for datasets that use valid as test split
    actual_split = "valid" if split == "test" else split
    
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR','sentences_and_BBox_mscxr')

    dataset_configs = []
    
    try:
        # MIMIC-CXR reports
        print(f"MIMIC-CXR reports ({actual_split})")
        mimic_dataset_filtered = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=actual_split, 
            flag_img=False, flag_lab=False, 
            only_frontal=True, 
            filtered_reports_dir=filtered_reports_dir, 
            seed=0
        )
        print("Num samples = " + str(len(mimic_dataset_filtered)))
        dataset_configs.append({"dataset": mimic_dataset_filtered, "id_prefix": f"mimic-{split}"})
    except Exception as e:
        print(f"Skipping MIMIC-CXR reports ({actual_split}): {e}")

    try:
        # MIMIC-CXR classification
        print(f"MIMIC-CXR classif ({actual_split})")
        mimic_dataset_labels = MIMIC_Dataset_MM(
            datasetpath=datasetpath_mimic,
            split=actual_split, 
            flag_img=False, flag_lab=True, 
            only_frontal=True, 
            filtered_reports_dir=None, 
            classif=True,
            seed=0
        )
        print("Num samples = " + str(len(mimic_dataset_labels)))
        dataset_configs.append({"dataset": mimic_dataset_labels, "id_prefix": f"mimic-labels-{split}"})
    except Exception as e:
        print(f"Skipping MIMIC-CXR classification ({actual_split}): {e}")

    try:
        # CheXpert
        print(f"CheXpert classif ({actual_split})")
        dataset_path = os.path.join(data_dir, "CheXpert")   
        chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path, split=actual_split, flag_img=False)
        print("Num samples = " + str(len(chexpert_dataset)))
        dataset_configs.append({"dataset": chexpert_dataset, "id_prefix": f"chexpert-{split}"})
    except Exception as e:
        print(f"Skipping CheXpert ({actual_split}): {e}")

    # CheXpert-Plus only has training data
    if split == "train":
        try:
            print(f"CheXpert reports (train only)")
            datasetpath = os.path.join(data_dir, 'CheXpert')
            filtered_reports_dir = os.path.join(datasetpath, 'filtered_reports')
            chexpertplus_dataset = CheXpertPlus_Dataset(
                datasetpath=datasetpath, 
                split='train', 
                flag_img=False, 
                filtered_reports_dir=filtered_reports_dir
            )
            print("Num samples = " + str(len(chexpertplus_dataset)))
            dataset_configs.append({"dataset": chexpertplus_dataset, "id_prefix": "chexpertplus-train"})
        except Exception as e:
            print(f"Skipping CheXpert-Plus: {e}")

    # Chest-ImaGenome only has training data  
    if split == "train":
        try:
            print(f"Chest-ima (train only)")
            datasetpath_chestima = os.path.join(data_dir, 'CHEST_IMA')
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
            print("Num samples = " + str(len(chestima_dataset)))
            dataset_configs.append({"dataset": chestima_dataset, "id_prefix": "chestima-train", "num_samples": 80000})
        except Exception as e:
            print(f"Skipping Chest-ImaGenome: {e}")

    try:
        # VinDr-CXR
        print(f"VinDr-CXR ({actual_split})")
        dataset_path = os.path.join(data_dir, "VinDr-CXR") 
        vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split=actual_split, flag_img=False)
        print("Num samples = " + str(len(vin_dataset)))
        
        vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split=actual_split, flag_img=False)
        print("Mono samples = " + str(len(vin_dataset_mono)))
        
        # Add multiple copies as in original
        if split == "train":
            dataset_configs.extend([
                {"dataset": vin_dataset, "id_prefix": f"vindr-cxr-{split}1"},
                {"dataset": vin_dataset, "id_prefix": f"vindr-cxr-{split}2"},
                {"dataset": vin_dataset_mono, "id_prefix": f"vindr-cxr-mono-{split}1"},
                {"dataset": vin_dataset_mono, "id_prefix": f"vindr-cxr-mono-{split}2"},
                {"dataset": vin_dataset_mono, "id_prefix": f"vindr-cxr-mono-{split}3"},
            ])
        else:
            dataset_configs.extend([
                {"dataset": vin_dataset, "id_prefix": f"vindr-cxr-{split}"},
                {"dataset": vin_dataset_mono, "id_prefix": f"vindr-cxr-mono-{split}"},
            ])
    except Exception as e:
        print(f"Skipping VinDr-CXR ({actual_split}): {e}")

    try:
        # MS-CXR phrase grounding (train uses both train+valid, test uses only valid)
        print(f"Phrase grounding MS-CXR ({split})")
        if split == "train":
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
            phrase_grounding_mscxr_dataset = ConcatDataset([dataset_train, dataset_valid])
        else:  # test/valid
            phrase_grounding_mscxr_dataset = MS_CXR(
                datasetpath=datasetpath_mimic,
                split="valid", flag_img=False, 
                flag_lab=True, only_frontal=True, 
                flag_instr=True, 
                sentencesBBoxpath=sentencesBBoxpath,
                seed=0
            )
        
        print("Num samples = " + str(len(phrase_grounding_mscxr_dataset)))
        
        if split == "train":
            dataset_configs.extend([
                {"dataset": phrase_grounding_mscxr_dataset, "id_prefix": f"mscxr-{split}1"},
                {"dataset": phrase_grounding_mscxr_dataset, "id_prefix": f"mscxr-{split}2"},
                {"dataset": phrase_grounding_mscxr_dataset, "id_prefix": f"mscxr-{split}3"},
            ])
        else:
            dataset_configs.append({"dataset": phrase_grounding_mscxr_dataset, "id_prefix": f"mscxr-{split}"})
    except Exception as e:
        print(f"Skipping MS-CXR ({split}): {e}")

    try:
        # PadChest phrase grounding
        print(f"Phrase grounding PadChest ({split})")
        datasetpath = os.path.join(data_dir, 'PadChest')
        
        if split == "train":
            dataset_train = PadChest_grounding(
                datasetpath=datasetpath,
                split='train', 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
            dataset_valid = PadChest_grounding(
                datasetpath=datasetpath,
                split='valid', 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )
            phrase_grounding_padchest_dataset = ConcatDataset([dataset_train, dataset_valid])
        else:  # test/valid
            phrase_grounding_padchest_dataset = PadChest_grounding(
                datasetpath=datasetpath,
                split='valid', 
                flag_instr=True,
                flag_img=False,
                flag_txt=False
            )

        print("Num samples = " + str(len(phrase_grounding_padchest_dataset)))
        
        if split == "train":
            dataset_configs.extend([
                {"dataset": phrase_grounding_padchest_dataset, "id_prefix": f"padchest-{split}1"},
                {"dataset": phrase_grounding_padchest_dataset, "id_prefix": f"padchest-{split}2"},
            ])
        else:
            dataset_configs.append({"dataset": phrase_grounding_padchest_dataset, "id_prefix": f"padchest-{split}"})
    except Exception as e:
        print(f"Skipping PadChest grounding ({split}): {e}")

    # Conversations (check if directories exist)
    conversation_dir = os.path.join(datasetpath_mimic, f'conversations/{actual_split}/standard')
    if os.path.exists(conversation_dir):
        try:
            print(f"Conversations standard ({actual_split})")
            conv_dataset_standard = MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split=actual_split, 
                flag_img=False, 
                flag_instr=False, 
                flag_txt=False, 
                flag_lab=False, 
                filtered_reports_dir=filtered_reports_dir,
                conversation_dir=conversation_dir
            )
            print("Num samples = " + str(len(conv_dataset_standard)))
            dataset_configs.append({"dataset": conv_dataset_standard, "id_prefix": f"conv-{split}"})
        except Exception as e:
            print(f"Skipping standard conversations ({actual_split}): {e}")

    conversation_dir_grounded = os.path.join(datasetpath_mimic, f'conversations/{actual_split}/grounding')
    if os.path.exists(conversation_dir_grounded):
        try:
            print(f"Conversations grounded ({actual_split})")
            conv_dataset_grounded = MIMIC_Dataset_MM(
                datasetpath=datasetpath_mimic,
                split=actual_split, flag_img=False, 
                flag_lab=False, only_frontal=True, 
                flag_instr=False, 
                filtered_reports_dir=filtered_reports_dir,
                sentencesBBoxpath=sentencesBBoxpath,
                conversation_dir=conversation_dir_grounded,
                classif=False,
                seed=0
            )
            print("Num samples = " + str(len(conv_dataset_grounded)))
            
            if split == "train":
                dataset_configs.extend([
                    {"dataset": conv_dataset_grounded, "id_prefix": f"conv-grounded-{split}1"},
                    {"dataset": conv_dataset_grounded, "id_prefix": f"conv-grounded-{split}2"},
                    {"dataset": conv_dataset_grounded, "id_prefix": f"conv-grounded-{split}3"},
                    {"dataset": conv_dataset_grounded, "id_prefix": f"conv-grounded-{split}4"},
                ])
            else:
                dataset_configs.append({"dataset": conv_dataset_grounded, "id_prefix": f"conv-grounded-{split}"})
        except Exception as e:
            print(f"Skipping grounded conversations ({actual_split}): {e}")

    conversation_dir_pad = os.path.join(data_dir, f'PadChest/conversations/{actual_split}/grounding')
    if os.path.exists(conversation_dir_pad):
        try:
            print(f"Conversations grounded padchest ({actual_split})")
            dataset_train = PadChest_grounding_per_image(
                datasetpath=data_dir + '/PadChest',
                split=actual_split,
                flag_instr=False,
                flag_img=False,
                conversation_dir=conversation_dir_pad
            )
            print("Num samples = " + str(len(dataset_train)))
            
            if split == "train":
                dataset_configs.extend([
                    {"dataset": dataset_train, "id_prefix": f"conv-grounded-padchest-{split}1"},
                    {"dataset": dataset_train, "id_prefix": f"conv-grounded-padchest-{split}2"},
                    {"dataset": dataset_train, "id_prefix": f"conv-grounded-padchest-{split}3"},
                    {"dataset": dataset_train, "id_prefix": f"conv-grounded-padchest-{split}4"},
                ])
            else:
                dataset_configs.append({"dataset": dataset_train, "id_prefix": f"conv-grounded-padchest-{split}"})
        except Exception as e:
            print(f"Skipping PadChest grounded conversations ({actual_split}): {e}")

    return dataset_configs


def generate_vlmr1_dataset_from_instruction_dataset(dataset_info, base_dir, batch_size=64, num_workers=8, seed=0):
    """
    Generate VLM-R1 JSONL dataset from RadVLM datasets.
    Adapted from generate_llava_dataset_from_instruction_dataset.
    """
    vlmr1_samples = []

    for dataset_i, dataset_info_cell in enumerate(dataset_info):
        # Define DataLoader
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
    parser = argparse.ArgumentParser(description="Generate VLM-R1 JSONL datasets from RadVLM")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", default="./vlmr1_datasets", help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "both"], 
                       help="Data split to generate ('both' creates train and test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Generating VLM-R1 dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create the exact same dataset configurations as the original create_llava_dataset.py
    datasetpath_mimic = os.path.join(args.data_dir, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(args.data_dir, 'MIMIC-CXR-JPG/filtered_reports')
    sentencesBBoxpath = os.path.join(args.data_dir, 'MS-CXR','sentences_and_BBox_mscxr')

    # MIMIC-CXR reports
    print("MIMIC-CXR reports")
    mimic_dataset_filtered = MIMIC_Dataset_MM(
        datasetpath=datasetpath_mimic,
        split="train", 
        flag_img=False, flag_lab=False, 
        only_frontal=True, 
        filtered_reports_dir=filtered_reports_dir, 
        seed=0
    )
    print("Num samples = " + str(len(mimic_dataset_filtered)))

    # MIMIC-CXR classification
    print("MIMIC-CXR classif")
    mimic_dataset_labels = MIMIC_Dataset_MM(
        datasetpath=datasetpath_mimic,
        split="train", 
        flag_img=False, flag_lab=True, 
        only_frontal=True, 
        filtered_reports_dir=None, 
        classif=True,
        seed=0
    )
    print("Num samples = " + str(len(mimic_dataset_labels)))

    # CheXpert
    print("CheXpert classif")
    dataset_path = os.path.join(args.data_dir, "CheXpert")   
    chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path, split="train", flag_img=False)
    print("Num samples = " + str(len(chexpert_dataset)))

    # CheXpert-Plus reports
    print("CheXpert reports")
    datasetpath = os.path.join(args.data_dir, 'CheXpert')
    filtered_reports_dir = os.path.join(datasetpath, 'filtered_reports')
    chexpertplus_dataset = CheXpertPlus_Dataset(
        datasetpath=datasetpath, 
        split='train', 
        flag_img=False, 
        filtered_reports_dir=filtered_reports_dir
    )
    print("Num samples = " + str(len(chexpertplus_dataset)))

    # Chest-ImaGenome
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
    print("Num samples = " + str(len(chestima_dataset)))

    # VinDr-CXR
    print("VinDr-CXR")
    dataset_path = os.path.join(args.data_dir, "VinDr-CXR") 
    vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img=False)
    print("Num samples = " + str(len(vin_dataset)))

    vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="train", flag_img=False)
    print("Num samples = " + str(len(vin_dataset_mono)))

    # MS-CXR phrase grounding
    print("Phrase grounding MS-CXR")
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
    print("Num samples = " + str(len(prhase_grounding_mscxr_dataset)))

    # PadChest phrase grounding
    print("Phrase grounding PadChest")
    datasetpath = os.path.join(args.data_dir, 'PadChest')
    dataset_train = PadChest_grounding(
        datasetpath=datasetpath,
        split='train', 
        flag_instr=True,
        flag_img=False,
        flag_txt=False
    )

    dataset_valid = PadChest_grounding(
        datasetpath=datasetpath,
        split='valid', 
        flag_instr=True,
        flag_img=False,
        flag_txt=False
    )

    prhase_grounding_padchest_dataset = ConcatDataset([dataset_train, dataset_valid])
    print("Num samples = " + str(len(prhase_grounding_padchest_dataset)))

    # Conversations (if available)
    conv_dataset_standard = None
    conv_dataset_grounded = None
    conv_dataset_grounded_padchest = None

    conversation_dir = os.path.join(datasetpath_mimic, 'conversations/train/standard')
    if os.path.exists(conversation_dir):
        print("Conversations standard")
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
        print("Num samples = " + str(len(conv_dataset_standard)))

    conversation_dir_grounded = os.path.join(datasetpath_mimic, 'conversations/train/grounding')
    if os.path.exists(conversation_dir_grounded):
        print("Conversations grounded")
        conv_dataset_grounded = MIMIC_Dataset_MM(
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
        print("Num samples = " + str(len(conv_dataset_grounded)))

    conversation_dir_pad = os.path.join(args.data_dir, 'PadChest/conversations/train/grounding')
    if os.path.exists(conversation_dir_pad):
        print("Conversations grounded padchest")
        dataset_train = PadChest_grounding_per_image(
            datasetpath=args.data_dir + '/PadChest',
            split='train',
            flag_instr=False,
            flag_img=False,
            conversation_dir=conversation_dir_pad
        )
        dataset_valid = PadChest_grounding_per_image(
            datasetpath=args.data_dir + '/PadChest',
            split='valid',
            flag_instr=False,
            flag_img=False,
            conversation_dir=conversation_dir_pad
        )
        conv_dataset_grounded_padchest = ConcatDataset([dataset_train, dataset_valid])
        print("Num samples = " + str(len(conv_dataset_grounded_padchest)))

    # Create dataset_info list (same as original)
    dataset_info = [
        {"dataset": vin_dataset, "id_prefix": "vindr-cxr-train1"},
        {"dataset": vin_dataset, "id_prefix": "vindr-cxr-train2"},
        {"dataset": vin_dataset_mono, "id_prefix": "vindr-cxr-mono-train1"},
        {"dataset": vin_dataset_mono, "id_prefix": "vindr-cxr-mono-train2"},
        {"dataset": vin_dataset_mono, "id_prefix": "vindr-cxr-mono-train3"},
        {"dataset": prhase_grounding_mscxr_dataset, "id_prefix": "mscxr-train1"},
        {"dataset": prhase_grounding_mscxr_dataset, "id_prefix": "mscxr-train2"},
        {"dataset": prhase_grounding_mscxr_dataset, "id_prefix": "mscxr-train3"},
        {"dataset": prhase_grounding_padchest_dataset, "id_prefix": "padchest-train1"},
        {"dataset": prhase_grounding_padchest_dataset, "id_prefix": "padchest-train2"},
        {"dataset": mimic_dataset_filtered, "id_prefix": "mimic-train"},
        {"dataset": chexpertplus_dataset, "id_prefix": "chexpertplus-train"},
        {"dataset": chestima_dataset, "id_prefix": "chestima-train", "num_samples": 80000},
        {"dataset": mimic_dataset_labels, "id_prefix": "mimic-labels-train"},
        {"dataset": chexpert_dataset, "id_prefix": "chexpert-train"},
    ]

    # Add conversation datasets if they exist
    if conv_dataset_standard:
        dataset_info.append({"dataset": conv_dataset_standard, "id_prefix": "conv-train"})
    
    if conv_dataset_grounded:
        dataset_info.extend([
            {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train1"},
            {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train2"},
            {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train3"},
            {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train4"},
        ])
    
    if conv_dataset_grounded_padchest:
        dataset_info.extend([
            {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train1"},
            {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train2"},
            {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train3"},
            {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train4"},
        ])

    # Generate VLM-R1 dataset
    print(f"\nGenerating VLM-R1 dataset from {len(dataset_info)} dataset configurations...")
    vlmr1_dataset = generate_vlmr1_dataset_from_instruction_dataset(
        dataset_info, 
        args.data_dir,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        seed=args.seed
    )

    # Save as JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    
    with open(output_path, "w", encoding='utf-8') as f:
        for sample in vlmr1_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ VLM-R1 dataset saved!")
    print(f"📁 File: {output_path}")
    print(f"📊 Total samples: {len(vlmr1_dataset):,}")
    
    print(f"\n🚀 Usage with VLM-R1:")
    print(f"python -m open_r1.grpo_jsonl \\")
    print(f"    --data_file_paths {output_path} \\")
    print(f"    --image_folders {args.data_dir}/ \\")
    print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")


if __name__ == "__main__":
    main()