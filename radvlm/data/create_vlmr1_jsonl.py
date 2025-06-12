#!/usr/bin/env python3
#!/usr/bin/env python3
import os, sys, json, argparse, random, glob

# ensure the project root is on PYTHONPATH so `import radvlm` works
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from radvlm.data.create_instructions import generate_instruction_phrase_location
from radvlm.data.utils import custom_collate_fn
from torch.utils.data import DataLoader


def format_boxes(bboxes, num_float=2):
    strs = [f"[{round(x1,num_float)}, {round(y1,num_float)}, {round(x2,num_float)}, {round(y2,num_float)}]"
            for x1, y1, x2, y2 in bboxes]
    if not strs:
        return "[]"
    if len(strs) == 1:
        return strs[0]
    return ", ".join(strs[:-1]) + " and " + strs[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM-R1 JSONL from MS-CXR phrase-grounding annotations"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root of your public_radiology_repo (contains MS-CXR folder)"
    )
    parser.add_argument(
        "--out-file", default="train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle the order of examples"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (for shuffle)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # locate MS-CXR directories
    ms_root = os.path.join(args.data_dir, "MS-CXR")
    json_dir = os.path.join(ms_root, "sentences_and_BBox_mscxr")
    img_root = os.path.join(ms_root, "images_grounding")

    # gather grounding JSON files
    json_paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if args.shuffle:
        random.shuffle(json_paths)
    print(f"Found {len(json_paths)} grounding samples.")

    # one-time index of all images
    print(f"Indexing all JPGs under {img_root}…", end='', flush=True)
    image_paths = glob.glob(os.path.join(img_root, "**", "*.jpg"), recursive=True)
    id2img = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    print(f" done ({len(image_paths)} images).")

    # write JSONL
    with open(args.out_file, 'w') as fout:
        for i, js in enumerate(json_paths):
            data = json.load(open(js, 'r'))
            # extract boxes & phrase
            boxes = [e['box'] for e in data]
            phrase = data[0].get('observation', '')

            # instruction
            instr = generate_instruction_phrase_location(boxes, phrase)

            # image lookup
            uuid = os.path.splitext(os.path.basename(js))[0]
            full_img = id2img.get(uuid)
            if full_img is None:
                raise FileNotFoundError(f"No image found for ID {uuid}")

            # make image path relative to --data-dir
            rel_img = os.path.relpath(full_img, args.data_dir)

            # build cell
            cell = {
                "id": f"mscXR-train_{i}",
                "image": rel_img,
                "conversations": [
                    {"from": "human", "value": f"<image>{instr['question']}"},
                    {"from": "gpt",   "value": instr['answer']}
                ]
            }
            fout.write(json.dumps(cell, ensure_ascii=False) + "\n")

            # progress
            if (i + 1) % 100 == 0:
                print(f"  • {i+1}/{len(json_paths)} examples written")

    print(f"✅ Wrote {len(json_paths)} examples to {args.out_file}")


if __name__ == '__main__':
    main()

