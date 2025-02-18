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

| Task                    | Dataset source    | Image-instruction pairs (#) | Evaluation (#) | DUA                                                                                                                                          |
|-------------------------|-------------------|-----------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Report Generation       | MIMIC-CXR         | 232,344 × 1                 | 3,282          | [Yes](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert-Plus     | 178,368 × 1                 | -              | [Yes](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1)                                                   |
| Abnormality classif.    | MIMIC-CXR         | 237,912 × 1                 | 518            | [Yes](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
|                         | CheXpert          | 191,027 × 1                 | -              | [Yes](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2)                                                   |
| Anatomical grounding    | Chest Imagenome   | 80,000 × 1                  | 2,000          | [Yes](https://physionet.org/content/chest-imagenome/1.0.0/)                                                                                    |
| Abnormality grounding   | VinDr-CXR         | 16,089 × 3                  | 2,108          | [Yes](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Abnormality detection   | VinDr-CXR         | 15,000 × 2                  | -              | [Yes](https://physionet.org/content/vindr-cxr/1.0.0/)                                                                                         |
| Phrase grounding        | MS-CXR            | 971 × 3                     | 189            | [Yes](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 4,478 × 2                   | -              | [Yes](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/)                                                  |
| Conversation            | MIMIC-CXR         | 80,312 × 1                  | 523            | [Yes](https://www.physionet.org/content/mimic-cxr-jpg/2.1.0/)                                                                                 |
| Conversation (grounded) | MS-CXR            | 858 × 4                     | 157            | [Yes](https://physionet.org/content/ms-cxr/0.1/)                                                                                              |
|                         | PadChest-GR       | 2,225 × 4                   | -              | [Yes](https://bimcv.cipf.es/bimcv-projects/padchest/padchest-dataset-research-use-agreement/)                                                  |

