# TallyQA

> [TallyQA GitHub repository](https://github.com/manoja328/TallyQA_dataset?tab=readme-ov-file)

As of Nov. 2018, TallyQA is the **largest open-ended counting dataset** for VQA. It is also the only dataset to distinguish between simple and complex counting questions. In summary, it has

- 287K questions
- 165K images
- 19K complex questions collected from human annotators using AMT

## Initialization

1. Create `eval/datasets/TallyQA/images` directory and change to it:
```bash
mkdir -p eval/datasets/TallyQA/images
cd eval/datasets/TallyQA/images
```
2. Download and unzip the images:
```bash
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
unzip images.zip

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images2.zip

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```