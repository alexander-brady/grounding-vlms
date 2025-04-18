# Evaluation Datasets

This folder contains data for four counting‐focused vision benchmarks. 

---

## Data Format & Usage

Each metadata file (`dataset.csv`) uses one of two row formats:

- `file_name,prompt,truth` if storing images locally
- `image_url,prompt,truth` if storing images remotely

where:
- `file_name` is the local path to the image starting from the image folder (e.g. `VG_100K/0001.png`)
- `image_url` is the URL to the image
- `prompt` is the question for the model (e.g. “How many giraffes are in the image?”)
- `truth` is the integer answer to the question (e.g. “3”)

For the PixMo dataset we also include an `image_hash` column to verify downloads.

---

## 1. GeckoNum Benchmark  
https://github.com/google-deepmind/geckonum_benchmark_t2i  
- **Content**: AI‑generated (non‑realistic) images with text prompts and human‑annotated integer answers.  
- **Why use it**: Its synthetic style differs from all other real‑world counting datasets.  
- **Notes**: We filtered out entries with ambiguous or non‐integer answers (e.g. “10+”, “4.5”), keeping only clear integer labels. We further merged entries that were answered multiple times by different annotators. 
- **Storage**: Images are referenced by URL.

## 2. TallyQA  
https://github.com/manoja328/TallyQA_dataset  
- **Content**: Real images (COCO & Visual Genome) paired with counting questions that range from simple (“How many giraffes?”) to multi‐step reasoning (“How many giraffes are drinking water?”).
- **Why we use it**: It includes a variety of counting questions, from simple to complex, and is based on real images. 
- **Storage**: Images are stored locally in the `images` folder.

## 3. PixMo Count  
https://huggingface.co/datasets/allenai/pixmo-count   
- **Content**: Focused on the 2–10 object count range.
- **Notes**: Includes an image hash per entry to detect corrupted downloads.  
- **Storage**: Images are referenced by URL.

## 4. FSC‑147  
https://github.com/cvlab-stonybrook/LearningToCountEverything  
- **Content**: High‑density scenes with an average of ~56 objects per image, human‑annotated counts.  
- **Why we use it**: Stresses model performance on high‐count scenarios beyond typical low‐count benchmarks.
- **Storage**: Images are stored locally in the `images` folder.