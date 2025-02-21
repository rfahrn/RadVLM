import os
from PIL import Image
import numpy as np
import torch
import re
import argparse

from radvlm.data.utils import  process_sbb
from radvlm.data.datasets import  MIMIC_Dataset_MM
from radvlm.evaluation.models_loading_inference import load_model_and_processor, inference_radialog, inference_llavaov, inference_llavamed
from radvlm.data.utils import inference_gpt4o_with_retry
from radvlm import DATA_DIR


parser = argparse.ArgumentParser(description="A script to evaluate conversations with GPT-4o.")
parser.add_argument("--api_key", type=str, required=True,
                        help="Your OpenAI API key.")
parser.add_argument("--grounding", action="store_true",
                    help="Set this flag to evaluate grounded conversations")
parser.add_argument('--model_name', type=str, default='radialog', help="The VLM to evaluate")
args = parser.parse_args()

os.environ['OPENAI_API_KEY'] = args.api_key
print("Using OpenAI API key from argument.")

script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(script_dir, "results")

split = "test"
datasetpath = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG')
filtered_reports_dir = os.path.join(DATA_DIR, 'MIMIC-CXR-JPG/filtered_reports_new')

if args.grounding:
    sentencesBBoxpath = os.path.join(DATA_DIR, 'MS-CXR/sentences_and_BBox_mscxr')
    conversation_dir= os.path.join(datasetpath, 'conversations/test/grounding')
else:
    sentencesBBoxpath = None
    conversation_dir= os.path.join(datasetpath, 'conversations/test/standard')

input_dataset = MIMIC_Dataset_MM(
    datasetpath = datasetpath,
    split="test", flag_img=False, 
    flag_lab=True, only_frontal=True, 
    flag_instr=True, 
    filtered_reports_dir=filtered_reports_dir,
    sentencesBBoxpath = sentencesBBoxpath,
    conversation_dir=conversation_dir,
    classif=False,
    seed=0)


print(len(input_dataset))

prefix_file_path = os.path.join(script_dir, 'prefix_evaluate_conv.txt')
with open(prefix_file_path, 'r') as file:
        prefix_content = file.read()

tokenizer, model, processor = load_model_and_processor(args.model_name, device_map='auto')

# Initialize a list to store scores
scores = []

for i in range(len(input_dataset)):
    imgpath = input_dataset[i]['img_path']
    image_id = os.path.splitext(os.path.basename(imgpath))[0]

    report = input_dataset[i]['txt']
    image = input_dataset[i]['img']
    # bounding_boxes = input_dataset[i]['boxes']
    sentencesBBox = input_dataset[i]['sentencesBBox']
    labels = input_dataset[i]['labels']
    report = input_dataset[i]['txt']
    view = input_dataset[i]['view']
    gt_conversation = input_dataset[i]['conversation']

    prompt = prefix_content + "List of Abnormalities: " + ", ".join(labels) + "\n"
    prompt = prompt + "Radiology report: " + report + "\n"
    prompt = prompt + "View: " + str(view) + "\n"
    if sentencesBBox is not None:
        processed_sbb = process_sbb(sentencesBBox)
        if processed_sbb is not None:
            prompt = prompt + "Selected observations with bounding boxes coordinates\n" + processed_sbb + "\n"
    prompt = prompt + "Here is the conversation to evaluate: " + "\n\n"
    
    chat_history = [] 
    try:
        for j in range(len(gt_conversation)):
            if gt_conversation[j]["from"] == "human":
                question = gt_conversation[j]["value"]
                prompt = prompt + "User: " + question + "\n"
            else:
                expected_answer = gt_conversation[j]["value"]
                prompt = prompt + "Expected answer: " + expected_answer + "\n"

                # Generate response from the model 
                image = Image.open(imgpath)
                with torch.no_grad():
                    if args.model_name == 'radialog':
                        response, chat_history = inference_radialog(tokenizer, model, imgpath, question, chat_history)
                    elif args.model_name == 'llavamed':
                        response, chat_history = inference_llavamed(model, processor, imgpath, question, chat_history)
                    else:
                        response, chat_history = inference_llavaov(model, processor, imgpath, question, chat_history)
                prompt = prompt + "Generated answer: " + response + "\n\n"

    except Exception as e:
        # Log the error and skip the current item in the main loop
        print(f"Error during inference at dataset index {i}: {e}")
        continue
    
    prompt = prompt + "Note: write the overall score (/10) this way, so I can extract it: Overall score: <score>" + "\n"

    generated_text = inference_gpt4o_with_retry(prompt, model='gpt-4o')

    print("-------------------------------------\n\n\n\n")

    print(prompt)
    print(imgpath)
    print(generated_text)
    # Use regular expression to capture the score after "Overall score: "
    match = re.search(r'Overall score:\s*([\d\.]+)', generated_text)
    
    if match:
        try:
            score = float(match.group(1))
            scores.append(score)
        except ValueError:
            # Ignore non-numeric outputs
            pass
    print(scores)
    average_score = np.mean(scores)
    print(f"RUNNING AVERAGE SCORE: {average_score}")

    average_score_file = os.path.join(OUTPUT_DIR, "average_score.txt")
    with open(average_score_file, "w") as f:
        f.write(str(average_score))


print("EVALUATION COMPLETED")









    





    
    
    

    

        
    



