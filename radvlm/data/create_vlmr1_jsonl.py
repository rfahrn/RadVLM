#!/usr/bin/env python
import os
import sys
# — ensure your project root is on PYTHONPATH so `import radvlm` works
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import random
from glob import glob
import numpy as np
from torch.utils.data import DataLoader  # not really used here, but kept for parity
from radvlm.data.create_instructions import generate_instruction_phrase_location
from radvlm.data.utils import custom_collate_fn  # not used, but safe to import

def main():
    p = argparse.ArgumentParser(
        description="Generate VLM-R1 JSONL from MS-CXR BBox files"
    )
    p.add_argument("--data-dir", required=True,
                   help="root of public_radiology_repo (containing MS-CXR/)")
    p.add_argument("--out-file", default="train_scxr.jsonl",
                   help="where to write the JSONL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle the order of samples")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ms_root = os.path.join(args.data_dir, "MS-CXR")
    bbox_glob = os.path.join(ms_root, "sentences_and_BBox_mscxr", "*.json")
    json_files = sorted(glob(bbox_glob))
    if args.shuffle:
        random.shuffle(json_files)

    # build a quick map uuid → full image path
    img_root = os.path.join(ms_root, "images_grounding")
    image_map = {}
    for dirpath, _, files in os.walk(img_root):
        for fn in files:
            if fn.lower().endswith((".jpg", ".png")):
                uuid = os.path.splitext(fn)[0]
                image_map[uuid] = os.path.join(dirpath, fn)

    print(f"Found {len(json_files)} JSON samples; {len(image_map)} images indexed.")

    wrote = 0
    with open(args.out_file, "w") as fout:
        for i, js in enumerate(json_files):
            uuid = os.path.splitext(os.path.basename(js))[0]
            img_full = image_map.get(uuid)
            if img_full is None:
                sys.stderr.write(f"⚠️  no image found for ID {uuid}, skipping\n")
                continue

            # relative path under --image_folders
            img_rel = os.path.relpath(img_full, args.data_dir)

            data = json.load(open(js, "r"))
            boxes = [e["box"] for e in data]
            phrase = data[0].get("observation", "") if data else ""

            instr = generate_instruction_phrase_location(boxes, phrase)
            cell = {
                "id": f"mscXR-train_{wrote}",
                "image": img_rel,
                "conversations": [
                    {"from": "human", "value": f"<image>{instr['question']}"},
                    {"from": "gpt",   "value": instr["answer"]}
                ]
            }

            fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"✅ Wrote {wrote} samples to {args.out_file}")

if __name__ == "__main__":
    main()

    print("✅ Done.")
