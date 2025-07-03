
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
    parser = argparse.ArgumentParser(description="Generate VLM-R1 JSONL datasets from RadVLM (exact same as LLaVA but JSONL format)")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Root data directory")
    parser.add_argument("--output-dir", default="/capstor/scratch/cscs/rfahrni/vlmr1_datasets", help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"], 
                       help="Data split to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Generating VLM-R1 dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate training data (exactly like the original create_llava_dataset.py)
    if args.split in ["train", "both"]:
        print(f"\n=== GENERATING TRAINING DATA (same as create_llava_dataset.py) ===")
        
        # Copy the exact logic from create_llava_dataset.py
        datasetpath_mimic = os.path.join(args.data_dir, 'MIMIC-CXR-JPG')
        filtered_reports_dir = os.path.join(args.data_dir, 'MIMIC-CXR-JPG/filtered_reports')

        # MIMIC-CXR reports
        print("MIMIC-CXR reports")
        mimic_dataset_filtered = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                                  split="train", 
                                                  flag_img=False, flag_lab=False, 
                                                  only_frontal=True, 
                                                  filtered_reports_dir=filtered_reports_dir, 
                                                  seed=0
                                                 )
        print("Num samples = " + str(len(mimic_dataset_filtered)))

        # MIMIC-CXR classification
        print("MIMIC-CXR classif")
        mimic_dataset_labels = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
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
        chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path,split="train", flag_img=False)
        print("Num samples = " + str(len(chexpert_dataset)))

        # CheXpert-Plus reports
        print("CheXpert reports")
        datasetpath = os.path.join(args.data_dir, 'CheXpert')
        filtered_reports_dir_chex = os.path.join(datasetpath, 'filtered_reports')
        chexpertplus_dataset = CheXpertPlus_Dataset(datasetpath=datasetpath, split='train', flag_img=False, filtered_reports_dir=filtered_reports_dir_chex)
        print("Num samples = " + str(len(chexpertplus_dataset)))

        # CHEST_IMA
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
        vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
        print("Num samples = " + str(len(vin_dataset)))

        vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path, split="train", flag_img = False)
        print("Num samples = " + str(len(vin_dataset_mono)))

        # Phrase grounding MS-CXR
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
        print("Num samples = " + str(len(prhase_grounding_mscxr_dataset)))

        # Phrase grounding PadChest
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
        print("Num samples = " + str(len(prhase_grounding_padchest_dataset)))

        # CONVERSATIONS
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
            print("Num samples = " + str(len(conv_dataset_standard)))
        else:
            print("Conversations standard directory not found - skipping")

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
            print("Num samples = " + str(len(conv_dataset_grounded)))
        else:
            print("Conversations grounded directory not found - skipping")

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
            print("Num samples = " + str(len(conv_dataset_grounded_padchest)))
        else:
            print("PadChest conversations grounded directory not found - skipping")

        # Create dataset_info list (exactly like original)
        dataset_info = [
            {"dataset":vin_dataset, "id_prefix":"vindr-cxr-train1"},
            {"dataset":vin_dataset, "id_prefix":"vindr-cxr-train2"},
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train1"},
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train2"},
            {"dataset":vin_dataset_mono, "id_prefix":"vindr-cxr-mono-train3"},
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train1"},
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train2"},
            {"dataset":prhase_grounding_mscxr_dataset, "id_prefix":"mscxr-train3"},
            {"dataset":prhase_grounding_padchest_dataset, "id_prefix":"padchest-train1"},
            {"dataset":prhase_grounding_padchest_dataset, "id_prefix":"padchest-train2"},
            {"dataset":mimic_dataset_filtered, "id_prefix":"mimic-train"},
            {"dataset":chexpertplus_dataset, "id_prefix":"chexpertplus-train"},
            {"dataset":chestima_dataset, "id_prefix":"chestima-train", "num_samples":80000},
            {"dataset":mimic_dataset_labels, "id_prefix":"mimic-labels-train"},
            {"dataset":chexpert_dataset, "id_prefix":"chexpert-train"},
        ]

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

        # Generate VLM-R1 training dataset
        print(f"\nGenerating VLM-R1 training dataset from {len(dataset_info)} configurations...")
        train_vlmr1_dataset = generate_vlmr1_dataset_from_instruction_dataset(
            dataset_info, 
            args.data_dir,
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            seed=args.seed
        )

        # Save training data
        train_output_path = os.path.join(args.output_dir, "all_train.jsonl")
        with open(train_output_path, "w", encoding='utf-8') as f:
            for sample in train_vlmr1_dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"\n✅ Training dataset saved!")
        print(f"📁 File: {train_output_path}")
        print(f"📊 Training samples: {len(train_vlmr1_dataset):,}")
    
    # Generate test data (use valid split where available)
    if args.split in ["test", "both"]:
        print(f"\n=== GENERATING TEST DATA (using valid splits) ===")
        
        test_dataset_info = []
        
        # Only datasets that have valid/test splits
        try:
            print("MIMIC-CXR reports (test)")
            mimic_test = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                         split="valid", 
                                         flag_img=False, flag_lab=False, 
                                         only_frontal=True, 
                                         filtered_reports_dir=filtered_reports_dir, 
                                         seed=0)
            print("Num samples = " + str(len(mimic_test)))
            test_dataset_info.append({"dataset": mimic_test, "id_prefix": "mimic-test"})
        except Exception as e:
            print(f"Skipping MIMIC-CXR reports test: {e}")

        try:
            print("MIMIC-CXR classif (test)")
            mimic_labels_test = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic,
                                               split="valid", 
                                               flag_img=False, flag_lab=True, 
                                               only_frontal=True, 
                                               filtered_reports_dir=None, 
                                               classif=True,
                                               seed=0)
            print("Num samples = " + str(len(mimic_labels_test)))
            test_dataset_info.append({"dataset": mimic_labels_test, "id_prefix": "mimic-labels-test"})
        except Exception as e:
            print(f"Skipping MIMIC-CXR classif test: {e}")

        try:
            print("CheXpert classif (test)")
            chexpert_test = CheXpert_Dataset_MM(datasetpath=dataset_path, split="valid", flag_img=False)
            print("Num samples = " + str(len(chexpert_test)))
            test_dataset_info.append({"dataset": chexpert_test, "id_prefix": "chexpert-test"})
        except Exception as e:
            print(f"Skipping CheXpert test: {e}")

        try:
            print("VinDr-CXR (test)")
            vin_test = VinDr_CXR_Dataset(datasetpath=dataset_path, split="test", flag_img=False)
            print("Num samples = " + str(len(vin_test)))
            test_dataset_info.append({"dataset": vin_test, "id_prefix": "vindr-cxr-test"})
        except Exception as e:
            print(f"Skipping VinDr-CXR test: {e}")

        if test_dataset_info:
            print(f"\nGenerating VLM-R1 test dataset from {len(test_dataset_info)} configurations...")
            test_vlmr1_dataset = generate_vlmr1_dataset_from_instruction_dataset(
                test_dataset_info, 
                args.data_dir,
                batch_size=args.batch_size, 
                num_workers=args.num_workers, 
                seed=args.seed
            )

            # Save test data
            test_output_path = os.path.join(args.output_dir, "all_test.jsonl")
            with open(test_output_path, "w", encoding='utf-8') as f:
                for sample in test_vlmr1_dataset:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"\n✅ Test dataset saved!")
            print(f"📁 File: {test_output_path}")
            print(f"📊 Test samples: {len(test_vlmr1_dataset):,}")
        else:
            print("No test datasets available")

    if args.split == "train":
        print(f"\n🚀 Usage with VLM-R1:")
        print(f"python -m open_r1.grpo_jsonl \\")
        print(f"    --data_file_paths {train_output_path} \\")
        print(f"    --image_folders {args.data_dir}/ \\")
        print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
    elif args.split == "both":
        print(f"\n🚀 Usage with VLM-R1:")
        print(f"# Training:")
        print(f"python -m open_r1.grpo_jsonl \\")
        print(f"    --data_file_paths {train_output_path} \\")
        print(f"    --image_folders {args.data_dir}/ \\")
        print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")
        if 'test_output_path' in locals():
            print(f"\n# Testing:")
            print(f"python -m open_r1.grpo_jsonl \\")
            print(f"    --data_file_paths {test_output_path} \\")
            print(f"    --image_folders {args.data_dir}/ \\")
            print(f"    --model_name \"Qwen/Qwen2.5-VL-7B-Instruct\"")


if __name__ == "__main__":
    main()
