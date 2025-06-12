#!/usr/bin/env python3
#!/usr/bin/env python3
import os, sys, json, argparse, random, glob
from radvlm.data.create_instructions import generate_instruction_phrase_location
from radvlm.data.utils import custom_collate_fn
from torch.utils.data import DataLoader#!/usr/bin/env python
import os
import sys
import json
import argparse
import random
from glob import glob
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn
from radvlm.data.create_instructions import generate_instruction_phrase_location

def format_boxes(bboxes, numf=2):
    fmt = [f"[{round(x1,numf)}, {round(y1,numf)}, {round(x2,numf)}, {round(y2,numf)}]"
           for x1,y1,x2,y2 in bboxes]
    if not fmt: return "[]"
    if len(fmt)==1: return fmt[0]
    return ", ".join(fmt[:-1]) + " and " + fmt[-1]

def build_image_index(image_root):
    idx = {}
    for root,_,files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(".jpg") or f.lower().endswith(".png"):
                idx[f] = os.path.join(root, f)
    return idx

def main():
    p = argparse.ArgumentParser(description="Create VLM-R1 JSONL from MS-CXR grounding")
    p.add_argument("--data-dir",    required=True,
                   help="root of public_radiology_repo")
    p.add_argument("--out-file",    required=True,
                   help="where to write train.jsonl")
    p.add_argument("--shuffle",     action="store_true",
                   help="shuffle examples")
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    MS_ROOT       = os.path.join(args.data_dir, "MS-CXR")
    GLOB_PATTERN  = os.path.join(MS_ROOT, "sentences_and_BBox_mscxr", "*.json")
    IMG_INDEX_ROOT= os.path.join(args.data_dir, "MIMIC-CXR-JPG", "files")

    # 1) find grounding jsons
    all_js = sorted(glob(GLOB_PATTERN))
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(all_js)
    print(f"Found {len(all_js)} grounding samples.")

    # 2) index all real images once
    print(f"Indexing all images under {IMG_INDEX_ROOT} …", end="", flush=True)
    img_index = build_image_index(IMG_INDEX_ROOT)
    print(f" done ({len(img_index)} files).")

    # 3) stream out JSONL
    with open(args.out_file, "w") as fout:
        for i, js in enumerate(all_js):
            data = json.load(open(js))
            uuid  = os.path.splitext(os.path.basename(js))[0]
            img_fn= uuid + ".jpg"
            if img_fn not in img_index:
                raise FileNotFoundError(f"No image for ID {uuid} → looked for {img_fn}")
            full_img = img_index[img_fn]
            # make path relative to data_dir
            rel_img = os.path.relpath(full_img, args.data_dir)

            # conv
            boxes  = [e["box"] for e in data]
            phrase = data[0].get("observation","")
            instr  = generate_instruction_phrase_location(boxes, phrase)

            cell = {
                "id": f"mscXR-train_{i}",
                "image": rel_img,
                "conversations": [
                    {"from":"human", "value":f"<image>{instr['question']}"},
                    {"from":"gpt",   "value": instr["answer"]}
                ]
            }
            fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
            if (i+1) % 100 == 0:
                print(f"  • wrote {i+1}/{len(all_js)} lines")

    print("✅ Done.")
