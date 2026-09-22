# Dual-Branch Latent Adaptive Fusion for Multimodal Autism Spectrum Disorder Screening
## Usage
The training and testing experiments are implemented with PyTorch and tested on NVIDIA RTX GPUs.
### 1. Preparation
> The project is developed using VSCode.
- Install required dependencies:
```bash
pip install -r Requirements.txt
```
### 2. Datasets
> Place raw and preprocessed dataset under your dataset folder. Annotation index files define sample-label mapping for 5-fold cross validation.
### 3. Training & Evaluation Config
> The script `train.py` implements the full 5-fold subject-independent stratified cross-validation pipeline.
Running `train.py` will sequentially train, validate and test the model on all 5 folds automatically.
The final evaluation metrics are averaged across the 5 folds to report the overall performance.

- Run the training and evaluation script:
```bash
python train.py
```