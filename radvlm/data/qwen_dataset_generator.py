import random
import numpy as np
import json
import os
from collections import defaultdict
from collections import Counter
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn


def format_boxes(bounding_boxes, num_float=2):
    formatted_boxes = [
        f"[{round(bbox[0], num_float)}, {round(bbox[1], num_float)}, {round(bbox[2], num_float)}, {round(bbox[3], num_float)}]"
        for bbox in bounding_boxes
    ]
    boxes_str = (
        ", ".join(formatted_boxes[:-1])
        + ("" if len(formatted_boxes) < 2 else " and ")
        + formatted_boxes[-1]
    )
    return boxes_str


def select_article(word):
    if word[0].lower() in "aeiou":
        return "an"
    else:
        return "a"
    

def create_json_cell_qwen(sample, sample_idx, dataset):
    """
    Create a JSON cell in Qwen-2.5VL format instead of LLaVA format.
    
    Qwen-2.5VL format:
    {
        "id": 1,
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>What number of purple metallic balls are there?"},
            {"from": "gpt", "value": "0"}
        ]
    }
    """
    sample_instr = sample["conversation"] if "conversation" in sample else sample["instr"]
    json_cell = {
        "id": sample_idx,  # Numeric ID for Qwen
        "image": sample["img_path"],
        "conversations": []
    }

    if isinstance(sample_instr, dict):
        # Case when sample_instr is a dict with question and answer
        sample_instr = [sample_instr]  # Convert to a list for uniform processing
    
    for j, instr in enumerate(sample_instr):
        if "from" in instr and "value" in instr:
            # Case when it's already in the correct format
            conv_cell = instr.copy()
            if j == 0 and "<image>" not in conv_cell["value"]:
                # Add <image> token at the beginning for the first human message
                conv_cell["value"] = f"<image>{conv_cell['value']}"
            json_cell["conversations"].append(conv_cell)
        elif "question" in instr and "answer" in instr:
            # Case when it's a dict with question and answer keys
            conv_cell_human = {
                "from": "human",
                "value": instr["question"]
            }
            if j == 0:
                # Add <image> token at the beginning for the first human message
                conv_cell_human["value"] = f"<image>{conv_cell_human['value']}"
            conv_cell_ai = {
                "from": "gpt",
                "value": instr["answer"]
            }
            json_cell["conversations"].append(conv_cell_human)
            json_cell["conversations"].append(conv_cell_ai)
        else:
            continue

    # Optional: Add labels and pathologies as metadata (not required for Qwen format)
    if "labels" in sample:
        json_cell["labels"] = sample["labels"]
    if getattr(dataset, "pathologies", None):
        json_cell["pathologies"] = dataset.pathologies.copy()

    return json_cell


def generate_qwen_dataset_from_instruction_dataset(dataset_info, output_file, batch_size=64, num_workers=128, seed=0):
    """
    Generate structured data for Qwen-2.5VL format and save as JSONL.
    
    Args:
        dataset_info: List of dataset configuration dictionaries
        output_file: Path to output JSONL file
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        seed: Random seed
    """
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_samples = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for dataset_i, dataset_info_cell in enumerate(dataset_info):
            # Define DataLoader
            dataset = dataset_info_cell["dataset"]
            id_prefix = dataset_info_cell.get("id_prefix", f"dataset_{dataset_i}")
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
                    
                    json_cell = create_json_cell_qwen(sample, total_samples, dataset)
                    
                    # Write each JSON object as a line (JSONL format)
                    f.write(json.dumps(json_cell, ensure_ascii=False) + '\n')
                    
                    sample_count += 1
                    total_samples += 1

                if sample_count >= num_samples:
                    break
    
    print(f"Generated {total_samples} samples in JSONL format: {output_file}")
    return total_samples


# Keep all the instruction generation functions from the original code
def generate_instruction_report_generation(text, german_suffixe=False):
    """
    Generate a 1-round instruction (question and answer) for creating a radiology report
    from an X-ray/image, with a preference for selecting 'no preamble' in the answer most of the time.
    """
    questions = [
        "Provide a radiology report for this Chest X-Ray.",
        "Analyze the Chest X-Ray and report your findings.",
        "Please give a detailed radiology report from this Chest X-Ray image.",
        "Please offer a detailed analysis based on this image.",
        "Analyze this image and provide a report.",
        "Write a radiology report for this Chest X-ray.",
        "Draft a concise report for this image.",
        "Please provide a brief radiology report.",
        "Compose a radiology report focusing on key findings.",
        "Create a structured radiology report for this X-ray.",
        "Produce a summary radiology report for this chest image.",
        "Provide a radiology report for this exam.",
        "Write a report summarizing this Chest X-ray.",
        "Draft a simple radiology report for these findings.",
        "Outline a radiology report for this X-ray."
    ]

    if german_suffixe:
        questions = [q + " in German" for q in questions]

    question = random.choice(questions)
    instruction = {"question": question, "answer": f"{text}"}
    return instruction


def generate_instruction_phrase_location(bounding_boxes, label):
    question_variations = [
        "Please locate the following sentence: {}",
        "Identify the position of the following phrase in the CXR: {}",
        "Please show me the location of: {}",
        "Highlight the area of the following observation on the image: {}",
        "Where on the image can you see the following observation: {}",
        "Find the region corresponding to: {}",
        "Please indicate where this finding is located: {}",
        "Mark the area where you observe: {}"
    ]

    answer_variations = [
        "This sentence is located at the coordinates {} on the image.",
        "You'll find it at {} in the CXR.",
        "This phrase can be observed at {} on the image.",
        "The bounding box for this observation is {}.",
        "Its location is given by {}.",
        "It is displayed at {} in the radiograph.",
        "The area specified is at coordinates {}.",
        "This finding is located at {} in the image."
    ]
    
    boxes_str = format_boxes(bounding_boxes)

    if label[0].isupper() and not label.isupper():
        label = label.lower()
    question = random.choice(question_variations).format(label)
    answer = random.choice(answer_variations).format(boxes_str)

    instruction = {"question": question, "answer": answer}
    return instruction


def generate_instruction_location(bounding_boxes, label):
    """Generate location-based instructions for anatomical regions."""
    questions_variations = [
        "Where is the {} located in this Chest X-ray?",
        "Can you point out the {}'s position on the image?",
        "What's the location of the {} in the X-ray?",
        "Identify where the {} is on this Chest X-ray, please.",
        "Where exactly is the {} found on this image?",
        "Could you specify where to find the {} on this X-ray?",
        "Highlight the {}'s area on the image.",
        "Show me the {}'s location on this CXR.",
        "Where should I look to find the {} in this image?",
        "Can you locate the {} on this X-ray for me?"
    ]

    answer_variations = [
        "The {} is located at the coordinates {} on the image.",
        "You'll find the {} at {} in the X-ray.",
        "The {} can be seen at {} on the Chest X-ray.",
        "The location of the {} is at {} on the image.",
        "For the {}, the coordinates are {} on the X-ray.",
        "The {} is situated at {} in the image.",
        "On the Chest X-ray, the {} is located at {}.",
        "The {} appears at the coordinates {} on the image."
    ]

    boxes_str = format_boxes(bounding_boxes)

    if label[0].isupper() and not label.isupper():
        label = label.lower()
    question = random.choice(questions_variations).format(label)
    answer = random.choice(answer_variations).format(label, boxes_str)

    instruction = {"question": question, "answer": answer}
    return instruction


def generate_instruction_abnormalities_grouped(bounding_boxes, abnormalities):
    """Generate instructions for abnormality detection with grouping."""
    question_variations = [
        "Could you indicate if there are any abnormalities on this Chest X-ray and their locations?",
        "Are abnormalities present on this Chest X-ray? Where exactly can they be found?",
        "Please identify any lesions or abnormalities on this X-ray and specify their locations.",
        "On this Chest X-ray, can you point out any abnormalities and their precise positions?",
        "Can you detect and describe the location of any abnormalities found on this Chest X-ray?"
    ]

    answer_prefix_variations = [
        "Sure! I can find",
        "Indeed, there are",
        "Yes, the following abnormalities are identified:",
        "Upon examination, I detect",
        "The analysis reveals",
        "The findings include"
    ]

    no_lesions_answers = [
        "I can't find any lesion on the image.",
        "No abnormalities or lesions are detected on this Chest X-ray.",
        "The Chest X-ray appears to be clear of any lesions or abnormalities.",
        "Upon review, no lesions are visible on the image.",
        "This Chest X-ray shows no signs of abnormalities or lesions."
    ]

    question = random.choice(question_variations)

    if not bounding_boxes or not abnormalities:
        answer = random.choice(no_lesions_answers)
    else:
        if len(bounding_boxes) != len(abnormalities):
            raise ValueError("Bounding boxes and abnormalities lists must be of equal length.")

        abnormalities_dict = defaultdict(list)
        for abnormality, bbox in zip(abnormalities, bounding_boxes):
            abnormalities_dict[abnormality].append(bbox)

        abnormalities_descriptions = []
        for abnormality, boxes in abnormalities_dict.items():
            article = select_article(abnormality)
            abnormality = abnormality.lower()
            boxes_str = format_boxes(boxes)
            abnormalities_descriptions.append(
                f"{article} {abnormality} located at the coordinates {boxes_str}"
            )

        abnormalities_str = "; ".join(abnormalities_descriptions)
        answer_prefix = random.choice(answer_prefix_variations)
        answer = f"{answer_prefix} {abnormalities_str}."

    instruction = {"question": question, "answer": answer}
    return instruction


def generate_instruction_abnormalities(abnormalities):
    """Generate instructions for abnormality classification."""
    question_variations = [
        "Can you tell me if there are any abnormalities on this image?",
        "Are there any abnormalities on this Chest X-ray?",
        "Please identify abnormalities on this X-ray.",
        "Can you point out if there are any abnormalities on this Chest X-ray?",
        "Tell me about abnormalities on this image",
        "Do any abnormalities appear on this Chest X-ray?"
    ]

    if not abnormalities:
        no_abnormalities_answers = [
            "No abnormalities or lesions are detected on this Chest X-ray.",
            "The Chest X-ray appears to be clear of any abnormalities.",
            "There are no detectable abnormalities on this X-ray.",
            "This Chest X-ray shows no signs of abnormalities.",
            "Based on this X-ray, there are no abnormalities to report."
        ]
        answer = random.choice(no_abnormalities_answers)
    else:
        abnormality_count = Counter(abnormalities)
        abnormalities_list = [
            f"{abnormality.lower()}" for abnormality, _ in abnormality_count.items()
        ]

        if len(abnormalities_list) > 1:
            abnormalities_str = (
                ", ".join(abnormalities_list[:-1]) + " and " + abnormalities_list[-1]
            )
            answer_variations = [
                "The following abnormalities are identified:",
                "The following abnormalities are present:",
                "The analysis reveals these abnormalities:"
            ]
        else:
            abnormalities_str = abnormalities_list[0]
            answer_variations = [
                "Yes, there is one abnormality identified:",
                "The following abnormality is present:",
                "The analysis reveals the following abnormality:"
            ]

        answer_prefix = random.choice(answer_variations)
        answer = f"{answer_prefix} {abnormalities_str}."

    question = random.choice(question_variations)
    instruction = {"question": question, "answer": answer}
    return instruction


def generate_instruction_foreign_objects(bounding_boxes):
    """
    Generate a question and an answer about the location of potential foreign objects
    on a Chest X-ray, incorporating variations for questions and answers.
    """
    question_variations = [
        "Could you indicate if there are any foreign objects on this Chest X-ray and their locations?",
        "Are there any foreign objects visible on this Chest X-ray and, if so, where?",
        "Can you detect and pinpoint the location of any foreign objects on this X-ray?",
        "Do you see any foreign objects present on this Chest X-ray? Where exactly?",
        "Are any unexpected items or foreign objects showing up in this X-ray image?",
        "Can you identify if there are foreign objects in this X-ray, and provide their locations?",
        "Could you locate and describe any foreign objects in this Chest X-ray?",
        "On this X-ray, are there signs of any foreign objects? Where can they be found?",
        "Please point out any foreign objects on this X-ray along with their coordinates.",
        "Can you report the presence and locations of any foreign objects in this Chest X-ray?"
    ]

    answer_prefix_variations = [
        "Upon review, I have identified foreign objects located at the following coordinates:",
        "The X-ray reveals foreign objects at:",
        "Foreign objects are detected at these positions:",
        "I've located potential foreign objects in the X-ray at:",
        "The analysis indicates the presence of foreign objects at:",
        "Foreign objects are visible in the X-ray at:",
        "The following foreign objects have been found at:",
        "Detectable foreign objects are present at:",
        "The image analysis shows foreign objects located at:",
        "Foreign objects have been pinpointed at the following coordinates:"
    ]

    no_objects_answers = [
        "No foreign objects are detected on this Chest X-ray.",
        "The Chest X-ray is clear of any foreign objects.",
        "There are no detectable foreign objects in this X-ray image.",
        "I find no evidence of foreign objects on this X-ray.",
        "The X-ray does not show any foreign objects.",
        "No unexpected items or foreign objects are visible in the X-ray.",
        "There are no signs of foreign objects on this X-ray.",
        "The X-ray appears clean, with no foreign objects present.",
        "No foreign objects can be seen in the X-ray image.",
        "There's nothing that resembles foreign objects in this X-ray."
    ]

    question = random.choice(question_variations)

    if len(bounding_boxes) == 0:
        answer = random.choice(no_objects_answers)
    else:
        boxes_str = format_boxes(bounding_boxes)
        answer_prefix = random.choice(answer_prefix_variations)
        answer = f"{answer_prefix} {boxes_str}."

    instruction = {"question": question, "answer": answer}
    return instruction


# Example usage function
def create_qwen_dataset_example():
    """
    Example of how to use the modified functions to create a Qwen-2.5VL dataset.
    """
    # This is a template - you'll need to replace with your actual datasets
    dataset_info = [
        {
            "dataset": your_mimic_dataset,  # Replace with actual dataset
            "id_prefix": "mimic_cxr",
            "num_samples": 10000  # Optional: limit number of samples
        },
        {
            "dataset": your_chexpert_dataset,  # Replace with actual dataset  
            "id_prefix": "chexpert",
            "num_samples": 5000
        }
    ]
    
    # Generate the dataset in JSONL format
    output_file = "datasets/qwen_instruction_dataset.jsonl"
    total_samples = generate_qwen_dataset_from_instruction_dataset(
        dataset_info=dataset_info,
        output_file=output_file,
        batch_size=64,
        num_workers=128,
        seed=42
    )
    
    print(f"Successfully generated {total_samples} samples for Qwen-2.5VL format!")
    return output_file
