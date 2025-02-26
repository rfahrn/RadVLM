import argparse
import json
import os
from torch.utils.data import random_split
from multiprocessing import Pool
import torch

from radvlm.data.datasets import *
from radvlm.data.utils import process_sbb, inference_gpt4o_with_retry
from radvlm import DATA_DIR


def create_conversation_dataset(input_dataset, prefix_file_path, output_dir, model):
    with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(len(input_dataset)):
        # Stop if output directory already has 150,000 files
        if len(os.listdir(output_dir)) >= 100000:
            print("Reached 150,000 files. Interrupting the loop.")
            break

        imgpath = input_dataset[i]['img_path']
        image_id = os.path.splitext(os.path.basename(imgpath))[0]
        output_file_path = os.path.join(output_dir, f'{image_id}.json')
        if os.path.exists(output_file_path):
            print("Already done, skip!")
            continue

        report = input_dataset[i]['txt']
        sentencesBBox = input_dataset[i]['sentencesBBox']
        view = input_dataset[i]['view']
        
        # Build prompt
        prompt = prefix_content + "Radiology report: " + report + "\n"
        prompt += "View: " + str(view) + "\n"
        if sentencesBBox is not None and process_sbb(sentencesBBox) != '':
            processed_sbb = process_sbb(sentencesBBox)
            prompt += "Selected observations with bounding boxes coordinates:\n" + processed_sbb + "\n"
        
        prompt += "\nConversation in expected format:\n"
        print(prompt)
        
        generated_text = inference_gpt4o_with_retry(prompt, model=model)
        print(generated_text)
        print("--------------------------------------")

        # Extract JSON content from the generated text
        try:
            start_idx = generated_text.index("[")
            end_idx = generated_text.rindex("]") + 1
            extracted_content = generated_text[start_idx:end_idx]
            extracted_list = json.loads(extracted_content)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Could not extract a valid JSON list: {e}")
            extracted_list = None

        if isinstance(extracted_list, list):
            with open(output_file_path, 'w') as json_file:
                json.dump(extracted_list, json_file, indent=4)
                print("Output saved!")
        else:
            print("Could not extract a list")



def process_chunk(chunk_index, chunk, prefix_file_path, output_dir, api_key):
    print(f"Processing chunk {chunk_index} on process {os.getpid()}")
    create_conversation_dataset(chunk, prefix_file_path, output_dir)



def main():
    parser = argparse.ArgumentParser(
        description="Conversation dataset creation script with GPT4o inference."
    )
    parser.add_argument("--azure_model", type=str, required=True,
                        help="The azume model name (gpt-4o, gpt-4o-mini, etc.) used to generate conversations ")
    parser.add_argument("--split", choices=['train', 'test'], type=str, required=True,
                        help="The dataset split")
    parser.add_argument("--grounding", action="store_true",
                    help="Set this flag to generate grounded conversations.")
    parser.add_argument("--padchest", action="store_true",
                    help="Set this flag to generate conversations for padchest dataset.")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Number of chunks to split the dataset into (and number of parallel processes).")
    args = parser.parse_args()

    # File paths and dataset configuration (adjust as needed)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not args.padchest:
        datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
        filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports_new')
        datasetpath_mscxr = os.path.join(DATA_DIR, 'MS-CXR')
        if args.grounding:
            sentencesBBoxpath = os.path.join(datasetpath_mscxr, 'sentences_and_BBox_mscxr')
            prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_conv_grounding.txt')
            folder_name = 'grounding'
        else:
            sentencesBBoxpath = None # full MIMIC-CXR dataset
            prefix_file_path = os.path.join(script_dir, 'prefixes_prompts/prefix_conv.txt')
            folder_name = 'standard'

        split = args.split
        dataset = MIMIC_Dataset_MM(
            datasetpath=datasetpath,
            split=split,
            flag_img=False,
            flag_lab=True,
            only_frontal=True,
            flag_instr=False,
            filtered_reports_dir=filtered_reports_dir,
            sentencesBBoxpath=sentencesBBoxpath,
            classif=False,
            seed=0
        )
        print(f"Total dataset size: {len(dataset)}")
    else:
        datasetpath = os.path.join(DATA_DIR, 'PadChest')
        split = 'train' 
        dataset = PadChest_grounding_per_image(
            datasetpath=datasetpath,
            split=split, 
            flag_instr=False
        )



    output_dir = os.path.join(datasetpath, 'conversations', split, folder_name)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Ensure reproducibility
    torch.manual_seed(125)

    num_chunks = args.num_chunks
    dataset_size = len(dataset)
    chunk_size = dataset_size // num_chunks
    remaining = dataset_size % num_chunks
    split_sizes = [chunk_size + 1 if i < remaining else chunk_size for i in range(num_chunks)]
    chunks = random_split(dataset, split_sizes)

    with Pool(processes=num_chunks) as pool:
        pool.starmap(
            process_chunk,
            [(i, chunks[i], prefix_file_path, output_dir, args.azure_model) for i in range(num_chunks)]
        )


if __name__ == "__main__":
    main()
