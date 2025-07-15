
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

def format_prompt_for_verl(sample):
    """
    Formats the instruction or conversation from a RadVLM sample into
    the list-of-dictionaries format required by VeRL's chat template.
    Maps 'from' -> 'role' and 'value' -> 'content'.
    """
    # Instruction can be under 'instr' or 'conversation' key
    instruction_data = sample.get('instr') or sample.get('conversation')

    if not instruction_data:
        return None

    # Standardize the instruction data into a list of turns
    if isinstance(instruction_data, dict):
        # Handles simple {question: ..., answer: ...} format
        turns = [
            {'from': 'human', 'value': instruction_data.get('question', '')},
            {'from': 'gpt', 'value': instruction_data.get('answer', '')}
        ]
    elif isinstance(instruction_data, list):
        # Handles pre-formatted conversational lists
        turns = instruction_data
    else:
        return None

    # Convert to VeRL's format
    verl_prompt = []
    for turn in turns:
        if not isinstance(turn, dict) or 'from' not in turn or 'value' not in turn:
            continue
            
        role = 'user' if turn['from'] == 'human' else 'assistant'
        # Remove the <image> token as it's not needed in the prompt itself for VeRL
        content = turn['value'].replace('<image>\n', '').strip()
        verl_prompt.append({'role': role, 'content': content})

    return verl_prompt


def create_verl_dataset():
    """
    Loads all RadVLM datasets, processes them, and saves them as a single
    Parquet file compatible with the VeRL framework.
    """
    print("Initializing datasets...")
    # --- Dataset Initialization (same as create_llava_dataset.py) ---
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
    phrase_grounding_mscxr_dataset = ConcatDataset([dataset_train_mscxr, dataset_valid_mscxr])

    datasetpath_padchest = os.path.join(DATA_DIR, 'PadChest')
    dataset_train_padchest = PadChest_grounding(datasetpath=datasetpath_padchest, split='train', flag_instr=True, flag_img=False, flag_txt=False)
    dataset_valid_padchest = PadChest_grounding(datasetpath=datasetpath_padchest, split='valid', flag_instr=True, flag_img=False, flag_txt=False)
    phrase_grounding_padchest_dataset = ConcatDataset([dataset_train_padchest, dataset_valid_padchest])

    conversation_dir_mimic_standard = os.path.join(datasetpath_mimic, 'conversations/train/standard')
    conv_dataset_standard = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_instr=False, flag_txt=False, flag_lab=False, filtered_reports_dir=filtered_reports_dir, conversation_dir=conversation_dir_mimic_standard)
    
    conversation_dir_mimic_grounding = os.path.join(datasetpath_mimic, 'conversations/train/grounding')
    conv_dataset_grounded = MIMIC_Dataset_MM(datasetpath=datasetpath_mimic, split="train", flag_img=False, flag_lab=False, only_frontal=True, flag_instr=False, filtered_reports_dir=filtered_reports_dir, sentencesBBoxpath=sentencesBBoxpath, conversation_dir=conversation_dir_mimic_grounding, classif=False, seed=0)

    conversation_dir_padchest_grounding = os.path.join(datasetpath_padchest, 'conversations/train/grounding')
    dataset_train_padchest_conv = PadChest_grounding_per_image(datasetpath=datasetpath_padchest, split='train', flag_instr=False, flag_img=False, conversation_dir=conversation_dir_padchest_grounding)
    dataset_valid_padchest_conv = PadChest_grounding_per_image(datasetpath=datasetpath_padchest, split='valid', flag_instr=False, flag_img=False, conversation_dir=conversation_dir_padchest_grounding)
    conv_dataset_grounded_padchest = ConcatDataset([dataset_train_padchest_conv, dataset_valid_padchest_conv])

    print("Combining all datasets...")
    combined_dataset = ConcatDataset([
        mimic_dataset_filtered, mimic_dataset_labels, chexpert_dataset, chexpertplus_dataset,
        chestima_dataset, vin_dataset, vin_dataset_mono, phrase_grounding_mscxr_dataset,
        phrase_grounding_padchest_dataset, conv_dataset_standard, conv_dataset_grounded,
        conv_dataset_grounded_padchest
    ])
    
    # Use a DataLoader for efficient processing
    data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    print(f"Processing {len(combined_dataset)} samples to create VeRL dataset...")
    verl_dataset = []
    # Using tqdm for a progress bar
    for sample_list in tqdm(data_loader, desc="Generating VeRL data"):
        sample = sample_list[0] # DataLoader with batch_size=1 returns a list
        
        # We are specifically interested in grounding tasks which have bounding boxes
        gt_boxes = sample.get('boxes')
        if not gt_boxes:
            continue

        # Format the prompt
        verl_prompt = format_prompt_for_verl(sample)
        if not verl_prompt:
            continue

        # Structure the data for VeRL
        output_sample = {
            "data_source": "RadVLM",
            "prompt": verl_prompt,
            "ability": "phrase_grounding",
            "reward_model": {
                "style": "iou",  # Specify IoU for reward calculation
                "ground_truth": gt_boxes
            },
            "extra_info": {
                "image_path": sample.get('img_path', ''),
                "dicom_id": os.path.basename(sample.get('img_path', '')).replace('.jpg', '')
            }
        }
        verl_dataset.append(output_sample)

    if not verl_dataset:
        print("Warning: No samples with bounding boxes were found. The output file will be empty.")
        return

    # --- Save to Parquet ---
    print(f"Saving {len(verl_dataset)} processed samples to Parquet file...")
    df = pd.DataFrame(verl_dataset)
    
    # Define the absolute path for the output file
    output_dir = 'C:/Users/rebec/OneDrive/Desktop/RL-finetuning/verl-main/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'radvlm_verl_train.parquet')
    
    df.to_parquet(output_path)
    print(f"VeRL dataset saved successfully at: {output_path}")

if __name__ == '__main__':
    create_verl_dataset()
