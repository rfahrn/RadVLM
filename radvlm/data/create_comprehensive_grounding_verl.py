import os
import argparse
import pandas as pd
from radvlm.data.create_verl_dataset_llava import generate_verl_dataset_from_info
from radvlm.data.datasets import (
    MS_CXR, VinDr_CXR_Dataset, VinDr_CXR_Single_Label_Dataset,
    PadChest_grounding, MIMIC_Dataset_MM
)
from radvlm import DATA_DIR
from torch.utils.data import ConcatDataset

def create_grounding_datasets(data_dir, output_dir, splits=['train', 'test']):
    """
    Create comprehensive grounding datasets using existing RadVLM infrastructure.
    Leverages the existing veRL dataset creation but focuses on grounding tasks.
    """
    
    # Paths
    datasetpath_mimic = os.path.join(data_dir, 'MIMIC-CXR-JPG')
    sentencesBBoxpath = os.path.join(data_dir, 'MS-CXR', 'sentences_and_BBox_mscxr')
    dataset_path_vindr = os.path.join(data_dir, "VinDr-CXR")
    datasetpath_padchest = os.path.join(data_dir, 'PadChest')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        print(f"\n=== Creating {split} dataset ===")
        
        # Initialize datasets for this split
        dataset_configs = []
        
        # MS-CXR (phrase grounding)
        if os.path.exists(sentencesBBoxpath):
            ms_cxr_dataset = MS_CXR(
                datasetpath=datasetpath_mimic,
                split=split,
                flag_img=False,
                flag_lab=True,
                only_frontal=True,
                flag_instr=True,
                sentencesBBoxpath=sentencesBBoxpath,
                seed=42
            )
            dataset_configs.append({
                "dataset": ms_cxr_dataset,
                "id_prefix": f"ms-cxr-{split}",
                "num_samples": len(ms_cxr_dataset)
            })
        
        # VinDr-CXR (abnormality grounding)
        if os.path.exists(dataset_path_vindr):
            vindr_dataset = VinDr_CXR_Dataset(
                datasetpath=dataset_path_vindr,
                split=split,
                flag_img=False
            )
            vindr_single_dataset = VinDr_CXR_Single_Label_Dataset(
                datasetpath=dataset_path_vindr,
                split=split,
                flag_img=False
            )
            
            dataset_configs.extend([
                {
                    "dataset": vindr_dataset,
                    "id_prefix": f"vindr-{split}",
                    "num_samples": len(vindr_dataset)
                },
                {
                    "dataset": vindr_single_dataset,
                    "id_prefix": f"vindr-single-{split}",
                    "num_samples": len(vindr_single_dataset)
                }
            ])
        
        # PadChest (grounding)
        if os.path.exists(datasetpath_padchest):
            # PadChest uses 'valid' instead of 'test'
            padchest_split = 'valid' if split == 'test' else split
            try:
                padchest_dataset = PadChest_grounding(
                    datasetpath=datasetpath_padchest,
                    split=padchest_split,
                    flag_instr=True,
                    flag_img=False,
                    flag_txt=False
                )
                dataset_configs.append({
                    "dataset": padchest_dataset,
                    "id_prefix": f"padchest-{split}",
                    "num_samples": len(padchest_dataset)
                })
            except Exception as e:
                print(f"Warning: Could not load PadChest {split} split: {e}")
        
        # Use existing veRL dataset generation
        if dataset_configs:
            print(f"Generating veRL dataset for {split} split...")
            verl_dataset = generate_verl_dataset_from_info(
                dataset_configs,
                batch_size=32,
                num_workers=4,
                seed=42
            )
            
            # Save to parquet
            if verl_dataset:
                df = pd.DataFrame(verl_dataset)
                output_path = os.path.join(output_dir, f'{split}.parquet')
                df.to_parquet(output_path)
                print(f"✅ {split} dataset saved: {output_path} ({len(verl_dataset)} samples)")
                
                # Print dataset statistics
                sources = df['data_source'].value_counts()
                print(f"Dataset sources for {split}:")
                for source, count in sources.items():
                    print(f"  {source}: {count} samples")
            else:
                print(f"❌ No samples generated for {split} split")

def main():
    parser = argparse.ArgumentParser(description="Create comprehensive grounding datasets for veRL using existing RadVLM infrastructure")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to datasets directory')
    parser.add_argument('--output_dir', type=str, default='./grounding_datasets', help='Output directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'], help='Dataset splits to create')
    
    args = parser.parse_args()
    
    # Set data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif 'DATA_DIR' in os.environ:
        data_dir = os.environ['DATA_DIR']
    else:
        raise ValueError("Please set DATA_DIR environment variable or provide --data_dir")
    
    create_grounding_datasets(data_dir, args.output_dir, args.splits)

if __name__ == '__main__':
    main()