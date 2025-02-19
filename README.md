# RadVLM

## Install dependencies
To install dependencies on the Swiss AI cluster, you can create a Docker file where you execute the following command"
```
conda create -n radvlm python=3.10 -y
conda activate radvlm
pip install -r requirements.txt
```

## Instruction dataset generation

### Dataset content
The instruction dataset comprises 1,022,742 image-instruction pairs spanning multiple vision-language tasks, including report generation, abnormality classification, anatomical and abnormality grounding, phrase grounding, and conversational interactions. Dataset sources and the corresponding number of image-instruction pairs are listed, with smaller datasets balanced by varying the frequency of instruction occurrences.

| Task                    | Dataset source    | Image-instruction pairs (#) | Evaluation (#) | DUA                                                                                                                                          |
|-------------------------|-------------------|-----------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Report Generation       | MIMIC-CXR         | 232,344 × 1                 | 3,282          | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert-Plus     | 178,368 × 1                 | -              | [stanfordaimi](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)                                                   |
| Abnormality classif.    | MIMIC-CXR         | 237,912 × 1                 | 518            | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert          | 191,027 × 1                 | -              | [stanfordaimi](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)                                                   |
| Anatomical grounding    | Chest Imagenome   | 80,000 × 1                  | 2,000          | [physionet](https://physionet.org/content/chest-imagenome/1.0.0/)                                                                                    |
| Abnormality grounding   | VinDr-CXR         | 16,089 × 3                  | 2,108          | [physionet](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Abnormality detection   | VinDr-CXR         | 15,000 × 2                  | -              | [physionet](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Phrase grounding        | MS-CXR            | 971 × 3                     | 189            | [physionet](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 4,478 × 2                   | -              | [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)                                                  |
| Conversation            | MIMIC-CXR         | 80,312 × 1                  | 523            | [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
| Conversation (grounded) | MS-CXR            | 858 × 4                     | 157            | [physionet](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 2,225 × 4                   | -              | [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest) / [bimcv](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)                                                  |

### Datasets download 

Each dataset can be downloaded via the links provided in the right column. Once the access is allowed, the datasets should be organized as follows: 
```
datasets/
├── MIMIC-CXR/
│   ├── mimic-cxr-2.0.0-chexpert.csv
│   ├── mimic-cxr-2.0.0-metadata.csv
│   ├── mimic-cxr-2.0.0-split.csv
│   ├── reports.csv * 
│   ├── files/
│   ├── filtered_reports/ *
│   └── conversations/ *
├── CheXpert/
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── CheXpertPlus/
│   ├── PNG/
│   │   ├── train/
│   │   └── valid/
│   ├── df_chexpert_plus_240401.csv
│   └── filtered_reports/ * 
├── CHEST_IMA/
│   └── silver_dataset/
├── VinDr-CXR/
│   ├── train_jpg/ * 
│   ├── test_jpg/ * 
│   ├── train/
│   ├── test/
│   ├── annotations_train.csv
│   ├── annotations_test.csv
│   ├── image_resolutions_train.json * 
│   └── image_resolutions_test.json * 
├── MS-CXR/
│   ├── MS_CXR_Local_Alignment_v1.0.0.csv
│   └── sentences_BBox_mscxr/ * 
└── PadChest/
    ├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
    ├── master_table.csv
    ├── grounded_reports_20240819.json
    └── images_grounding/
```
Make sure to set the environment variable `DATA_DIR` to the path of the main datasets directory. For example, if your datasets are located at `/home/username/datasets`, you can set the variable in your shell as follows:
```
export DATA_DIR=/home/username/datasets
```
In the above architecture, the files or folders marked with a `*` were not orginally part of the available datasets, and we describe below the procedure to generate each of them. The rest of the files are directly available in the official repositories. 

### Filtering reports in MIMIC-CXR and CheXpert-Plus
- The file `reports.csv` is obtained by following the findings/impression extraction procedure from the [official MIMIC-CXR github](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt). 
- The `filtered_reports` directory contains text reports filtered by the Azure OpenAI API call of GPT-4o. The reports are stored as txt files, organized by `study_id` (e.g., `53862424.txt`). In order to generate this directory, run the following command:
```
python data/llm_filter_reports.csv --api_key [azure_openAI_api_key] --split [train,test] --num_chunks [number of parallel API calls] 
```
This command will leverage the GPT-4o prompt stored in `data/prefixes_prompts/prefix_filter_reports.txt` to remove statements referring to previous studies. It should be executed for both `train` and `test` split values, in order to construct both `train` and `test` sets. 
Similarly, for CheXpertPlus, we can construct the `filtered_reports` folder, organized by studies, by executing the following command (only for train split):
```
python data/llm_filter_reports.csv --api_key [azure_openAI_api_key] --chexpertplus True --split train --num_chunks [number of parallel API calls] 
```

### Converting dicom to jpg in VinDr-CXR
The raw dataset of VinDr-CXR provides images in dicom format in folders `train` and `test`. To obtain the jpg images in directories `train_jpg` and `test_jpg`, as well as the files containing the image dimensions `image_resolutions_train.json` and `image_resolutions_test.json`, execute the following command:
```
python data/preprocess_scripts/dicom2jpg_vindrcxr.py
```

### Preprocess grounded phrases in MS-CXR
We re-organize the MS-CXR dataset by creating one json file per image (following MIMIC-CXR `image_id`), with bounding boxes normalized from 0 to 1. These are contained in the directory `sentences_BBox_mscxr/` that can be obtained by executing:
```
python data/preprocess_scripts/normalize_mscxr.py.py
```

### Generate conversations 
For MIMIC-CXR, in order to generate the `conversations` directory, we leverage GPT-4o by providing the corresponding prompt contained in `prefixes_prompts`, and execute the following command:
``` 
python data/llm_filter_reports.csv --api_key [azure_openAI_api_key] --padchest False --split [train,test] --grounding [True, False] --num_chunks [num API calls]
```
This should be performed for both train and test splits, each containing standard and grounded conversations. 
For PadChest-GR, just set the ` --padchest` argument to True, and only perform it for the train split and grounding argument. 
























