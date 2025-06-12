import os, sys

# make sure <PROJECT_ROOT> is on sys.path so that `import radvlm` works:
SCRIPT_DIR   = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# now imports below will find radvlm/
import json, argparse, random, numpy as np
from glob import glob
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn
from radvlm.data.create_instructions import generate_instruction_phrase_location
import sys, os, json, argparse, random
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from radvlm.data.utils import custom_collate_fn
from radvlm.data.create_instructions import generate_instruction_phrase_location

def format_boxes(bboxes, num_float=2):
    fmt = [f"[{round(x1,num_float)}, {round(y1,num_float)}, {round(x2,num_float)}, {round(y2,num_float)}]"
           for x1,y1,x2,y2 in bboxes]
    if not fmt: return "[]"
    if len(fmt)==1: return fmt[0]
    return ", ".join(fmt[:-1]) + " and " + fmt[-1]

def find_image(uuid, root):
    """Recursively glob for uuid.jpg under root/**."""
    pattern = os.path.join(root, "**", f"{uuid}.jpg")
    matches = glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find image for {uuid} under {root}")
    return matches[0]

def make_cell(js_path, data_dir, prefix, idx):
    # load boxes+phrase
    arr = json.load(open(js_path))
    boxes = [e["box"] for e in arr]
    phrase = arr[0]["observation"] if arr else ""
    # pick instruction
    instr = generate_instruction_phrase_location(boxes, phrase)

    # resolve image file
    uuid = os.path.splitext(os.path.basename(js_path))[0]
    img_full = find_image(uuid, os.path.join(data_dir, "MIMIC-CXR-JPG", "files"))
    # make it relative to data_dir
    img_rel  = os.path.relpath(img_full, data_dir)

    # build cell
    cell = {
      "id": f"{prefix}_{idx}",
      "image": img_rel,
      "conversations": [
         {"from":"human", "value": f"<image>{instr['question']}"},
         {"from":"gpt",   "value": instr['answer']}
      ]
    }
    return cell

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True,
                   help="Root of your public_radiology_repo")
    p.add_argument("--out-file",   default="train.jsonl")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers",type=int, default=8)
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ms_root = os.path.join(args.data_dir, "MS-CXR")
    ann_glob = os.path.join(ms_root, "sentences_and_BBox_mscxr", "*.json")
    js_files = sorted(glob(ann_glob))
    prefix   = "mscXR-train"

    # write out JSONL
    with open(args.out_file, "w") as fout:
        for idx, js in enumerate(js_files):
            cell = make_cell(js, args.data_dir, prefix, idx)
            fout.write(json.dumps(cell, ensure_ascii=False) + "\n")

    print(f"Wrote {len(js_files)} examples to {args.out_file}")

if __name__=="__main__":
    main()

