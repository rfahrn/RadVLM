#!/usr/bin/env python
import os
import json
import argparse
from torch.utils.data import DataLoader
from radvlm.data.datasets import MS_CXR
from radvlm.data.utils import custom_collate_fn

def create_grpo_rec_cell(sample, idx, data_root):
    img_rel = os.path.relpath(sample["img_path"], data_root)
    label  = sample["label"]
    boxes  = sample["boxes"]

    # 1) fixed prompt
    question = (
        "Please provide the bounding box coordinate "
        f"of the region this sentence describes: {label}"
    )

    # 2) JSON answer
    answer_obj = {"bbox_2d": boxes, "label": label}
    answer     = json.dumps(answer_obj, ensure_ascii=False)

    return {
        "id": idx,
        "image": img_rel,
        "conversations": [
            {"from": "human", "value": f"<image>{question}"},
            {"from": "gpt",   "value": answer}
        ]
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   required=True,
                   help="root of your image+annotations")
    p.add_argument("--out-file",   default="train_rec_grpo.jsonl")
    p.add_argument("--split",      default="train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers",type=int, default=8)
    p.add_argument("--shuffle",    action="store_true")
    args = p.parse_args()

    # load your REC dataset (here MS-CXR as an example)
    dataset = MS_CXR(
        datasetpath=os.path.join(args.data_dir, "MIMIC-CXR-JPG"),
        sentencesBBoxpath=os.path.join(args.data_dir, "MS-CXR", "sentences_and_BBox_mscxr"),
        split=args.split,
        flag_img=False, flag_instr=False, seed=0
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    with open(args.out_file, "w", encoding="utf-8") as fout:
        idx = 0
        for batch in loader:
            for sample in batch:
                cell = create_grpo_rec_cell(sample, idx, args.data_dir)
                fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
                idx += 1
    print(f"✅ Wrote {idx} examples to {args.out_file}")

if __name__ == "__main__":
    main()
