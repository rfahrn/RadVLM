#!/usr/bin/env python3
import os
import sys
import json
import glob
import random
import argparse

# ensure project root is on PYTHONPATH so imports resolve
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from radvlm.data.create_instructions import generate_instruction_phrase_location

def find_image(uuid: str, img_root: str) -> str:
    """
    Recursively glob for uuid.jpg under img_root/**.
    """
    pattern = os.path.join(img_root, "**", f"{uuid}.jpg")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find image for {uuid} under {img_root}")
    return matches[0]

def main():
    p = argparse.ArgumentParser(
        description="Generate VLM-R1 JSONL from MS-CXR phrase grounding annotations"
    )
    p.add_argument(
        "--data-dir", required=True,
        help="root of public_radiology_repo (contains MS-CXR and MIMIC-CXR-JPG)"
    )
    p.add_argument(
        "--out-file", default="train_scxr.jsonl",
        help="where to write the output JSONL"
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="random seed (for shuffling)"
    )
    p.add_argument(
        "--shuffle", action="store_true",
        help="shuffle the order of samples"
    )
    args = p.parse_args()

    ann_dir    = os.path.join(args.data_dir, "MS-CXR", "sentences_and_BBox_mscxr")
    img_root   = os.path.join(args.data_dir, "MIMIC-CXR-JPG", "files")
    json_paths = sorted(glob.glob(os.path.join(ann_dir, "*.json")))

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(json_paths)

    print(f"Found {len(json_paths)} grounding files → writing to {args.out_file}")
    with open(args.out_file, "w") as fout:
        for i, js_path in enumerate(json_paths):
            arr    = json.load(open(js_path))
            boxes  = [e["box"] for e in arr]
            phrase = arr[0].get("observation", "")

            instr = generate_instruction_phrase_location(boxes, phrase)

            # image lookup
            uuid     = os.path.splitext(os.path.basename(js_path))[0]
            img_full = find_image(uuid, img_root)
            img_rel  = os.path.relpath(img_full, args.data_dir)

            cell = {
                "id": f"mscXR-train_{i}",
                "image": img_rel,
                "conversations": [
                    {"from": "human", "value": f"<image>{instr['question']}"},
                    {"from": "gpt",   "value": instr['answer']}
                ]
            }

            fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
            # progress
            if (i + 1) % 100 == 0:
                print(f"  • {i+1}/{len(json_paths)} lines written")

    print("✅ Done!")

if __name__ == "__main__":
    main()
