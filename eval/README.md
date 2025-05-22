# Evaluation Datasets

This folder contains data for four counting‐focused vision benchmarks.

> Datasets were formatted with `datasets/clean.ipynb` (work in progress)

---

## Data Format & Usage

Each metadata file (`dataset.csv`) uses one of two row formats:

- `file_name,prompt,truth, label` if storing images locally
- `image_url,prompt,truth, label` if storing images remotely

where:
- `file_name` is the local path to the image starting from the image folder (e.g. `VG_100K/0001.png`)
- `image_url` is the URL to the image
- `prompt` is the question for the model (e.g. “How many giraffes are in the image?”)
- `truth` is the integer answer to the question (e.g. “3”)
- `label` is the label for the image (e.g. “giraffe”). These were either given in the original dataset or extracted from the prompt using a simple regex (if possible) or NLP.
---

## 1. FSC‑147  
https://github.com/cvlab-stonybrook/LearningToCountEverything  
- **Content**: High‑density scenes with an average of ~56 objects per image, human‑annotated counts.  
- **Why we use it**: Stresses model performance on high‐count scenarios beyond typical low‐count benchmarks.
- **Storage**: Images are stored locally in the `images` folder.

## 2. GeckoNum  
https://github.com/google-deepmind/geckonum_benchmark_t2i  
- **Content**: AI‑generated (non‑realistic) images with text prompts and human‑annotated integer answers.  
- **Why use it**: Its synthetic style differs from all other real‑world counting datasets.  
- **Notes**: We filtered out entries with ambiguous or non‐integer answers (e.g. “10+”, “4.5”), keeping only clear integer labels. We further merged entries that were answered multiple times by different annotators. 
- **Storage**: Images are referenced by URL.

## 3. TallyQA  
https://github.com/manoja328/TallyQA_dataset  
- **Content**: Real images (COCO & Visual Genome) paired with counting questions that range from simple (“How many giraffes?”) to multi‐step reasoning (“How many giraffes are drinking water?”).
- **Why we use it**: It includes a variety of counting questions, from simple to complex, and is based on real images. 
- **Storage**: Images are stored locally in the `images` folder.

## Analysis

Running the method `create_results` in `results.py` will create useful summaries and plots of the results.

This assumes the following folder structure:

```
├── eval
│   ├── valid_results
|   |  ├── model_name1
|   |  |  ├── dataset_name1_results.csv
|   |  |  ├── dataset2_results.csv
|   |  ├── model_name2
|   |  |  ├── dataset1_results.csv
|   |  |  ├── dataset2_results.csv
│   ├── data
│   |  ├── dataset_name1_dataset.csv
│   |  ├── dataset_name2_dataset.csv
├── src
│   ├── results.py
```