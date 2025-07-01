#!/usr/bin/env python
import os
import json
from torch.utils.data import DataLoader
from radvlm.data.datasets import MS_CXR
from radvlm.data.utils import custom_collate_fn

# ─── USER CONFIG ──────────────────────────────────────────────────────────────

# 1) Root of your public_radiology_repo (contains MIMIC-CXR-JPG and MS-CXR/…)
DATA_DIR       = "/capstor/store/cscs/swissai/a02/health_mm_llm_shared/data" #"/cluster/dataset/medinfmk/public_radiology_repo"

# 2) Where to write your output JSONLs (must be writable)
OUT_DIR        = "/capstor/scratch/cscs/rfahrni"  #"/cluster/home/fahrnr/rec_jsonl"

# 3) Which splits to export (must match MS_CXR.split names)
SPLITS         = ["train", "test"]

# 4) Dataloader params
BATCH_SIZE     = 64
NUM_WORKERS    = 8
SHUFFLE_TRAIN  = True   # whether to shuffle the 'train' split

# ─── END USER CONFIG ──────────────────────────────────────────────────────────

def create_grpo_rec_cell(sample, idx, data_root):
    # relative path from data_root to the image
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

def export_split(split_name):
    # prepare output path
    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f"{split_name}_rec_grpo.jsonl")

    # build the dataset
    dataset = MS_CXR(
        datasetpath=os.path.join(DATA_DIR, "MIMIC-CXR-JPG"),
        sentencesBBoxpath=os.path.join(DATA_DIR, "MS-CXR", "sentences_and_BBox_mscxr"),
        split=split_name,
        flag_img=False,
        flag_instr=False,
        seed=0
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(SHUFFLE_TRAIN if split_name=="train" else False),
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # write out the JSONL
    count = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for batch in loader:
            for sample in batch:
                cell = create_grpo_rec_cell(sample, count, DATA_DIR)
                fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
                count += 1

    print(f"✅ Wrote {count} examples to {out_file}")

def main():
    for split in SPLITS:
        export_split(split)

if __name__ == "__main__":
    main()
