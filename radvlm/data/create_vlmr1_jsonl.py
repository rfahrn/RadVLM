
import sys, os

# assume this file lives at <PROJECT_ROOT>/radvlm/data/create_vlmr1_jsonl.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import os
import json
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn

# ------------ Helper Functions ------------

def format_boxes(bounding_boxes, num_float=2):
    formatted = [
        f"[{round(b[0],num_float)}, {round(b[1],num_float)}, {round(b[2],num_float)}, {round(b[3],num_float)}]"
        for b in bounding_boxes
    ]
    if not formatted:
        return "[]"
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + " and " + formatted[-1]


def select_article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def as_rel(path: str, strip_root: str) -> str:
    return os.path.relpath(path, strip_root)




def create_json_cell_vlmr1(sample, id_prefix, idx, strip_root):
    # 1) Fix extension
    img = sample["img_path"]
    root, ext = os.path.splitext(img)
    if ext == "":
        if os.path.exists(root + ".jpg"):
            img = root + ".jpg"
        elif os.path.exists(root + ".png"):
            img = root + ".png"
    # 2) Make path relative for grpo_jsonl loader
    img = os.path.relpath(img, strip_root)

    # support both ChatML and question/answer dicts
    raw = sample.get("conversation", sample.get("instr"))
    cell = {
        "id": f"{id_prefix}_{idx}",
        "image": img,
        "conversations": []
    }

    # single QA dict
    if isinstance(raw, dict) and "question" in raw and "answer" in raw:
        q = raw["question"].strip()
        cell["conversations"].append({
            "from": "human",
            "value": f"<image>{q}"
        })
        a = raw["answer"].replace("<answer>", "").replace("</answer>", "").strip()
        cell["conversations"].append({
            "from": "gpt",
            "value": a
        })

    else:
        conv = raw if isinstance(raw, list) else [raw]
        for turn_idx, t in enumerate(conv):
            if not isinstance(t, dict):
                continue
            if "from" in t and "value" in t:
                role, value = t["from"], t["value"]
            else:
                role = "human" if "question" in t else "gpt"
                value = t.get("question", t.get("answer", ""))
            if role == "gpt":
                value = value.replace("<answer>", "").replace("</answer>", "").strip()
            if turn_idx == 0 and role == "human":
                value = f"<image>{value}"
            cell["conversations"].append({"from": role, "value": value})

    # optional metadata
    for k in ("labels", "pathologies"):
        if k in sample:
            cell[k] = sample[k]

    return cell


# ------------ Main Script ------------

def main():
    parser = argparse.ArgumentParser(description="Generate VLM-R1 JSONL from RadVLM datasets")
    parser.add_argument("--strip-root", type=str, required=True,
                        help="Root dir to strip from image paths (passed to --image_folders)")
    parser.add_argument("--out-file", type=str, default="train.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    # Expect dataset_info to be defined/imported below
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    #from radvlm.data.create_instructions import dataset_info  # Assumes dataset_info defined in that module
    #from radvlm.data.datasets import MS_CXR
    from glob import glob
    from radvlm.data.create_instructions import generate_instruction_phrase_location

    
    # Path to your on-disk MS-CXR folder:
    MS_ROOT = os.environ.get("DATA_DIR", "/cluster/dataset/medinfmk/public_radiology_repo") + "/MS-CXR"
    BBOX_GLOB = os.path.join(MS_ROOT, "sentences_and_BBox_mscxr", "*.json")
    
    json_files = sorted(glob(BBOX_GLOB))
    custom_dataset = []
    for j, js in enumerate(json_files):
        data = json.load(open(js, "r"))
        boxes = [entry["box"] for entry in data]
        phrase = data[0]["observation"] if data else ""
        # build the matching image path:
        img = os.path.join(
            MS_ROOT,
            "images_grounding",
            os.path.basename(js).replace(".json", ".jpg")
        )
        instr = generate_instruction_phrase_location(boxes, phrase)
        custom_dataset.append({"img_path": img, "instr": instr})

    dataset_info = [
        {
            "dataset": custom_dataset,
            "id_prefix": "mscXR-train",
            "num_samples": len(custom_dataset),
        }
    ]

    with open(args.out_file, "w") as fout:
        for block in dataset_info:
            ds = block["dataset"]
            prefix = block["id_prefix"]
            limit = block.get("num_samples", len(ds))
            loader = DataLoader(ds,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                collate_fn=custom_collate_fn)
            idx = 0
            for batch in loader:
                for sample in batch:
                    if idx >= limit: break
                    cell = create_json_cell_vlmr1(sample, prefix, idx, args.strip_root)
                    fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
                    idx += 1
                if idx >= limit:
                    break

    print(f"Wrote {args.out_file} with {idx} samples.")


if __name__ == '__main__':
    main()
    """ we now get: which is expected for VLM-R1
    {"id":"mscXR-train_0",
 "image":"/cluster/dataset/…/MS-CXR/images_grounding/672613f9-…jpg",
 "conversations":[
   {"from":"human","value":"<image>Please show me the location of: right apical pneumothorax"},
   {"from":"gpt","value":"This phrase can be observed at [0.22,0.19,0.47,0.27] on the image."}
 ]}
    """
