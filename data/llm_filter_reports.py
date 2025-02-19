import os
import re
import sys
import argparse
import json

import torch
import random
from torch.utils.data import random_split

# Import your GPT4o inference function from utils
from utils import inference_gpt4o_with_retry

# Example dataset classes – adjust these imports as needed
from datasets import MIMIC_Dataset_MM, CheXpertPlus_Dataset

###############################################################################
# Function to process one chunk of the dataset
###############################################################################
def extract_findings_for_chunk(input_chunk, prefix_file_path, output_dir, chexpertplus=False):
    """
    Processes a chunk of the dataset, extracting the findings for each sample
    and storing them in the specified output folder using GPT4o for inference.
    """
    # Read the prompt from the text file
    with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(input_chunk)):
        print("----------------------------------------------------")
        # Retrieve the image path
        imgpath = input_chunk[i]['img_path']

        # Name the output file by image_id or study_id
        if chexpertplus:
            txt_path = "_".join(imgpath.split('/')[-4:-1]) + ".txt"
            output_file_path = os.path.join(output_dir, txt_path)
        else:
            study_id = input_chunk[i].get("study_id", None)
            if study_id:
                output_file_path = os.path.join(output_dir, f"{study_id}.txt")
            else:
                image_id = os.path.splitext(os.path.basename(imgpath))[0]
                output_file_path = os.path.join(output_dir, f"{image_id}.txt")

        # Skip if already processed
        if os.path.exists(output_file_path):
            print(f"{output_file_path} already exists, skip!")
            continue

        # Get the report text
        report = input_chunk[i]['txt']
        if report is None:
            continue

        # Create the prompt
        prompt = prefix_content + report + "\n    - Extracted findings:\n"

        # Perform inference using GPT4o
        generated_text = inference_gpt4o_with_retry(prompt, model='gpt-4o')

        print("Generated text:")
        print(generated_text)

        # Save the generated text if it is valid
        if not generated_text or "None" in generated_text:
            print("Empty text or 'None' found; skipping save.")
        else:
            with open(output_file_path, 'w') as output_file:
                output_file.write(generated_text)

###############################################################################
# Worker function for processing a chunk in parallel
###############################################################################
def process_chunk(chunk_index, chunk, prefix_file_path, output_dir, api_key, chexpertplus):
    # Set the API key in this process
    os.environ['OPENAI_API_KEY'] = api_key
    print(f"Processing chunk {chunk_index} on process {os.getpid()}")
    extract_findings_for_chunk(chunk, prefix_file_path, output_dir, chexpertplus)

###############################################################################
# Main entry point, with multiprocessing-based chunking
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Filter reports script with GPT4o inference (parallel processing)."
    )
    parser.add_argument("--api_key", type=str, required=True,
                        help="Your OpenAI API key.")
    parser.add_argument("--chexpertplus", action="store_true",
                        help="If set, will process CheXpertPlus dataset logic (naming by image_id).")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="How many total chunks to split the dataset into (number of parallel processes).")
    args = parser.parse_args()

    # Set the API key in the main process (and it will be passed on to workers)
    os.environ['OPENAI_API_KEY'] = args.api_key
    print("Using OpenAI API key from argument.")

    DATA_DIR = os.environ.get('DATA_DIR')
    if DATA_DIR is None:
        raise EnvironmentError("The environment variable 'DATA_DIR' is not set.")

    # Adjust these paths as needed:
    # DATA_DIR = "/capstor/store/cscs/swissai/a02/health_mm_llm_shared/data"
    prefix_file_path = 'prefixes_prompts/prefix_filter_reports.txt'

    # Example for MIMIC-CXR:
    split = "train"
    if args.chexpertplus:
        # Use CheXpertPlus dataset logic
        datasetpath = os.path.join(DATA_DIR, 'CheXpertPlus')
        dataset = CheXpertPlus_Dataset(datasetpath=datasetpath, split=split, only_frontal=True, flag_img=False)
        output_dir = os.path.join(DATA_DIR, 'CheXpertPlus/filtered_reports')
    else:
        # Use MIMIC dataset by default
        datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
        dataset = MIMIC_Dataset_MM(
            datasetpath=datasetpath,
            split=split,
            filtered_reports_dir=None,
            flag_img=False,
            flag_lab=False,
            only_frontal=True
        )
        output_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports')

    print("Total dataset size:", len(dataset))
    torch.manual_seed(11)
    random.seed(11)

    # Split the dataset into chunks for parallel processing
    dataset_size = len(dataset)
    num_chunks = args.num_chunks
    chunk_size = dataset_size // num_chunks
    remainder = dataset_size % num_chunks
    split_sizes = [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)]
    chunks = random_split(dataset, split_sizes)

    # Use multiprocessing to process all chunks concurrently
    from multiprocessing import Pool
    with Pool(processes=num_chunks) as pool:
        pool.starmap(
            process_chunk,
            [(i, chunks[i], prefix_file_path, output_dir, args.api_key, args.chexpertplus)
             for i in range(num_chunks)]
        )

if __name__ == "__main__":
    main()
