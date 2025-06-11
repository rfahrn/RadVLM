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
    # Normalize conversation
    conv = sample.get("conversation", sample.get("instr"))
    if isinstance(conv, dict):
        conv = [conv]

    cell = {
        "id": f"{id_prefix}_{idx}",
        "image": as_rel(sample["img_path"], strip_root),
        "conversations": []
    }

    for turn_idx, t in enumerate(conv):
        # Determine role/value
        if "from" in t:
            role, value = t["from"], t["value"]
        else:
            role = "human" if "question" in t else "gpt"
            value = t.get("question", t.get("answer", ""))
        # Strip answer tags
        if role == "gpt":
            value = value.replace("<answer>", "").replace("</answer>", "").strip()
        # Prepend image tag on first human turn
        if turn_idx == 0 and role == "human":
            value = f"<image>{value.lstrip()}"
        cell["conversations"].append({"from": role, "value": value})
    # Optional metadata
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

    from radvlm.data.create_instructions import dataset_info  # Assumes dataset_info defined in that module

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
