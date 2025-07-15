import torch
import numpy as np
from radvlm.data.datasets import MIMIC_Dataset_MM, CheXpert_Dataset_MM, Chest_ImaGenome_Dataset, MS_CXR, CheXpertPlus_Dataset, PadChest_grounding, PadChest_grounding_per_image, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset
from radvlm.data.utils import custom_collate_fn
import json
import os
from torch.utils.data import ConcatDataset, DataLoader
from radvlm import DATA_DIR
import pandas as pd
from tqdm import tqdm
import random

def format_prompt_for_verl(sample):
    """
    Formats the instruction or conversation from a RadVLM sample into
    the list-of-dictionaries format required by VeRL's chat template.
    """
    instruction_data = sample.get('instr') or sample.get('conversation')
    if not instruction_data:
        return None

    if isinstance(instruction_data, dict):
        turns = [
            {'from': 'human', 'value': instruction_data.get('question', '')},
            {'from': 'gpt', 'value': instruction_data.get('answer', '')}
        ]
    elif isinstance(instruction_data, list):
        turns = instruction_data
    else:
        return None

    verl_prompt = []
    for turn in turns:
        if not isinstance(turn, dict) or 'from' not in turn or 'value' not in turn:
            continue
        role = 'user' if turn['from'] == 'human' else 'assistant'
        content = turn['value'].replace('<image>\n', '').strip()
        verl_prompt.append({'role': role, 'content': content})
    return verl_prompt

def generate_verl_dataset_from_info(dataset_info, batch_size=64, num_workers=0, seed=0):
    """
    Generates a VeRL-compatible dataset by processing a list of dataset configurations,
    mirroring the logic of `generate_llava_dataset_from_instruction_dataset`.
    """
    verl_structure = []
    np.random.seed(seed)
    random.seed(seed)

    for dataset_info_cell in dataset_info:
        dataset = dataset_info_cell["dataset"]
        id_prefix = dataset_info_cell["id_prefix"]
        num_samples = dataset_info_cell.get("num_samples", len(dataset))
        
        print(f"Processing {id_prefix} ({num_samples} samples)")

        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            collate_fn=custom_collate_fn
        )

        sample_count = 0
        progress_bar = tqdm(total=num_samples, desc=f"Processing {id_prefix}")
        for batch in data_loader:
            for sample in batch:
                if sample_count >= num_samples:
                    break

                gt_boxes = sample.get('boxes')
                if not gt_boxes:
                    continue

                verl_prompt = format_prompt_for_verl(sample)
                if not verl_prompt:
                    continue

                output_sample = {
                    "data_source": "RadVLM",
                    "prompt": verl_prompt,
                    "ability": "phrase_grounding",
                    "reward_model": {
                        "style": "iou",
                        "ground_truth": gt_boxes
                    },
                    "extra_info": {
                        "image_path": sample.get('img_path', ''),
                        "id_source": f"{id_prefix}_{len(verl_structure)}"
                    }
                }
                verl_structure.append(output_sample)
                sample_count += 1
                progress_bar.update(1)

            if sample_count >= num_samples:
                break
        progress_bar.close()
                
    return verl_structure

def main():
    """Main function to create the VeRL dataset."""
    if 'DATA_DIR' not in os.environ:
        os.environ['DATA_DIR'] = 'C:/Users/rebec/OneDrive/Desktop/RL-finetuning/RadVLM/RadVLM-main/datasets'
    
    print("Initializing all datasets...")
    # --- Dataset Initialization (1:1 copy from create_llava_dataset.py) ---
    datasetpath_mimic = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
    filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')
    mimic_dataset_filtered = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_lab=False, only_frontal=True, filtered_reports_dir=filtered_reports_dir, seed=0)
    mimic_dataset_labels = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_lab=True, only_frontal=True, filtered_reports_dir=None, classif=True, seed=0)
    dataset_path_chexpert = os.path.join(DATA_DIR, "CheXpert")
    chexpert_dataset = CheXpert_Dataset_MM(datasetpath=dataset_path_chexpert, split="train", flag_img=False)
    filtered_reports_dir_chexpert = os.path.join(dataset_path_chexpert, 'filtered_reports')
    chexpertplus_dataset = CheXpertPlus_Dataset(datasetpath=dataset_path_chexpert, split='train', flag_img=False, filtered_reports_dir=filtered_reports_dir_chexpert)
    datasetpath_chestima = os.path.join(DATA_DIR, 'CHEST_IMA')
    chestima_dataset = Chest_ImaGenome_Dataset(datasetpath=datasetpath_mimic, datasetpath_chestima=datasetpath_chestima, split="train", flag_img=False, flag_instr=True, flag_txt=False, flag_lab=False, pick_one_region=True)
    dataset_path_vindr = os.path.join(DATA_DIR, "VinDr-CXR")
    vin_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path_vindr, split="train", flag_img=False)
    vin_dataset_mono = VinDr_CXR_Single_Label_Dataset(datasetpath=dataset_path_vindr, split="train", flag_img=False)
    sentencesBBoxpath = os.path.join(DATA_DIR, 'MS-CXR', 'sentences_and_BBox_mscxr')
    dataset_train_mscxr = MS_CXR(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_lab=True, only_frontal=True, flag_instr=True, sentencesBBoxpath=sentencesBBoxpath, seed=0)
    dataset_valid_mscxr = MS_CXR(datasetpath=datasetpath_mimic, split="valid", flag_img=False, flag_lab=True, only_frontal=True, flag_instr=True, sentencesBBoxpath=sentencesBBoxpath, seed=0)
    prhase_grounding_mscxr_dataset = ConcatDataset([dataset_train_mscxr, dataset_valid_mscxr])
    datasetpath_padchest = os.path.join(DATA_DIR, 'PadChest')
    dataset_train_padchest = PadChest_grounding(datasetpath=datasetpath_padchest, split='train', flag_instr=True, flag_img=False, flag_txt=False)
    dataset_valid_padchest = PadChest_grounding(datasetpath=datasetpath_padchest, split='valid', flag_instr=True, flag_img=False, flag_txt=False)
    prhase_grounding_padchest_dataset = ConcatDataset([dataset_train_padchest, dataset_valid_padchest])
    conversation_dir_mimic_standard = os.path.join(datasetpath_mimic, 'conversations/train/standard')
    conv_dataset_standard = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_instr=False, flag_txt=False, flag_lab=False, filtered_reports_dir=filtered_reports_dir, conversation_dir=conversation_dir_mimic_standard)
    conversation_dir_mimic_grounding = os.path.join(datasetpath_mimic, 'conversations/train/grounding')
    conv_dataset_grounded = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_lab=False, only_frontal=True, flag_instr=False, filtered_reports_dir=filtered_reports_dir, sentencesBBoxpath=sentencesBBoxpath, conversation_dir=conversation_dir_mimic_grounding, classif=False, seed=0)
    conversation_dir_padchest_grounding = os.path.join(datasetpath_padchest, 'conversations/train/grounding')
    dataset_train_padchest_conv = PadChest_grounding_per_image(datasetpath=datasetpath_padchest, split='train', flag_instr=False, flag_img=False, conversation_dir=conversation_dir_padchest_grounding)
    dataset_valid_padchest_conv = PadChest_grounding_per_image(datasetpath=datasetpath_padchest, split='valid', flag_instr=False, flag_img=False, conversation_dir=conversation_dir_padchest_grounding)
    conv_dataset_grounded_padchest = ConcatDataset([dataset_train_padchest_conv, dataset_valid_padchest_conv])

    # --- dataset_info list (1:1 copy from create_llava_dataset.py) ---
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
        {"dataset": prhase_grounding_padchest_dataset, "id_prefix": "padchest-train3"},
        #{"dataset": mimic_dataset_filtered, "id_prefix": "mimic-train"},
        #{"dataset": chexpertplus_dataset, "id_prefix": "chexpertplus-train"},
        #{"dataset": chestima_dataset, "id_prefix": "chestima-train", "num_samples": 80000},
        #{"dataset": mimic_dataset_labels, "id_prefix": "mimic-labels-train"},
        #{"dataset": chexpert_dataset, "id_prefix": "chexpert-train"},
        #{"dataset": conv_dataset_standard, "id_prefix": "conv-train"},
        {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train1"},
        {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train2"},
        {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train3"},
        {"dataset": conv_dataset_grounded, "id_prefix": "conv-grounded-train4"},
        {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train1"},
        {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train2"},
        {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train3"},
        {"dataset": conv_dataset_grounded_padchest, "id_prefix": "conv-grounded-padchest-train4"},
    ]

    # --- Generate and Save VeRL Dataset ---
    verl_dataset = generate_verl_dataset_from_info(dataset_info)

    if not verl_dataset:
        print("Warning: No valid samples for VeRL were generated. The output file will be empty.")
        return

    print(f"Saving {len(verl_dataset)} processed samples to Parquet file...")
    df = pd.DataFrame(verl_dataset)
    output_dir = '/iopsstor/scratch/cscs/rfahrni/dataset/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'radvlm_verl_train_grounding.parquet')
    df.to_parquet(output_path)
    print(f"VeRL dataset saved successfully at: {output_path}")

if __name__ == '__main__':
    main()
