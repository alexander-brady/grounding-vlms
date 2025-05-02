# FSC-147 

> [FSC-147 GitHub repository](https://github.com/cvlab-stonybrook/LearningToCountEverything?tab=readme-ov-file)

FSC-147 is a counting dataset that contains 6135 images, spanning 147 different object categories such as kitchen utensils, office supplies, vehicles, and animals. The number of objects in each image ranges from 7 to 3731, with an average of 56 objects per image. The dataset is split into training, validation, and test sets. A total of 89 object categories are assigned to the training set, 29 to the validation set, and 29 to the test set, with different categories in each split. The training set  contains 3659 images, with the validation and test sets containing 1286 and 1190 images, respectively. For each image in the test set, a single category name is given, and the expected output is the number of instances.

## Initialization

1. Create `eval/datasets/FSC-147/images` directory and change to it:
```bash
mkdir -p eval/datasets/FSC-147/images
cd eval/datasets/FSC-147/images
```
2. Download the [dataset from Google Drive](`https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing`) and extract it, putting the files in the `eval/datasets/FSC-147/images` directory.
