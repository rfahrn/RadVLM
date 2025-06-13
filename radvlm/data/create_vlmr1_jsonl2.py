import os
import sys
# — ensure your project root is on PYTHONPATH so `import radvlm` works
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
#!/usr/bin/env python

#!/usr/bin/env python
import os
import sys
import json
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from radvlm.data.datasets import MS_CXR
from radvlm.data.utils import custom_collate_fn



def create_cell(sample, id_prefix, idx, data_dir):
    # sample["img_path"] is absolute; make it relative to data_dir
    img_rel = os.path.relpath(sample["img_path"], data_dir)
    # question from the generated instruction
    instr = sample.get("instr", {})
    question = instr.get("question", "")

    # build structured solution object with all boxes and the phrase label
    sol_obj = {
        "bbox": sample.get("boxes", []),
        "label": sample.get("label", "")
    }
    # dump JSON and wrap in <answer> tags
    answer = json.dumps(sol_obj, ensure_ascii=False)

    return {
        "id": f"{id_prefix}_{idx}",
        "image": img_rel,
        "conversations": [
            {"from": "human", "value": f"<image>{question}"},
            {"from": "gpt",   "value": f"<answer>{answer}</answer>"}
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM‑R1 JSONL with bbox+label for GRPO training"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="root of public_radiology_repo"
    )
    parser.add_argument(
        "--out-file", default="train_scxr.jsonl",
        help="where to write the JSONL"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle", action="store_true",
        help="shuffle the order of samples"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many samples"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mimic_jpg = os.path.join(args.data_dir, "MIMIC-CXR-JPG")
    bbox_jsons = os.path.join(args.data_dir, "MS-CXR", "sentences_and_BBox_mscxr")
    dataset = MS_CXR(
        datasetpath=mimic_jpg,
        sentencesBBoxpath=bbox_jsons,
        split="train",
        flag_img=False,
        flag_instr=True,
        seed=args.seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    count = 0
    with open(args.out_file, "w", encoding="utf-8") as fout:
        for batch in loader:
            for sample in batch:
                if args.limit and count >= args.limit:
                    break
                cell = create_cell(sample, "mscXR-train", count, args.data_dir)
                fout.write(json.dumps(cell, ensure_ascii=False) + "\n")
                count += 1
            if args.limit and count >= args.limit:
                break

    print(f"✅ Wrote {count} samples to {args.out_file}")


if __name__ == "__main__":
    main()
