# Skin Lesion Classification Using Swin Transformer

Deep learning project exploring Swin Transformer classification of eight skin lesion classes from the ISIC 2019 dataset.

## Overview

The project evaluates Swin-Tiny with three experimental techniques intended to reduce reliance on border artifacts:

- Border cropping and synthetic ruler augmentation
- Attention regularization
- Border-patch masking

The repository retains additional model implementations, while this documentation focuses on Swin experiments.

## Technologies

Python, PyTorch, torchvision, timm, NumPy, pandas, scikit-learn, Matplotlib, and Grad-CAM.

## Model

- Architecture: `swin_tiny_patch4_window7_224`
- Input size: 224 × 224 RGB images
- Output: eight diagnostic classes
- Initialization: `pretrained=False`
- Best checkpoint selection: validation macro recall

## Dataset

The project uses ISIC 2019 with the following diagnostic classes:

| Label | Class | Description |
|---|---|---|
| 0 | MEL | Melanoma |
| 1 | NV | Melanocytic nevus |
| 2 | BCC | Basal cell carcinoma |
| 3 | AK | Actinic keratosis |
| 4 | BKL | Benign keratosis |
| 5 | DF | Dermatofibroma |
| 6 | VASC | Vascular lesion |
| 7 | SCC | Squamous cell carcinoma |

The UNK class is excluded.

The setup script creates stratified image-level training, validation, and test splits of approximately 80%, 10%, and 10%, using seed 42.

Dataset source: https://www.kaggle.com/datasets/andrewmvd/isic-2019

Dataset images are downloaded separately and are not included in this repository.

## Swin Configurations

| Configuration | Description |
|---|---|
| Standard Swin | Swin-Tiny without an additional border-artifact technique |
| Technique 1 | Border cropping and synthetic ruler augmentation |
| Technique 2 | Attention regularization |
| Technique 3 | Border-patch masking |

## Results

These values come from the saved evaluation CSV files.

| Configuration | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|
| Standard Swin | 0.4048 | 0.5725 | 0.4342 |
| Technique 1 | 0.4104 | 0.5960 | 0.4471 |
| Technique 2 | 0.4003 | 0.5903 | 0.4215 |
| Technique 3 | 0.3731 | 0.5277 | 0.3850 |

Technique 1 achieved the highest macro recall and macro F1 among the four saved Swin runs. Technique 2 improved recall relative to standard Swin but reduced macro F1. Technique 3 reduced both metrics.

## Project Structure

- `dataset_setup.py`: dataset preparation and split generation
- `train.py`: model training and checkpoint saving
- `test.py`: checkpoint evaluation and output generation
- `requirements.txt`: Python dependencies
- `src/dataloader.py`: dataset loading and preprocessing
- `src/models/swin_transformer.py`: Swin-Tiny implementation
- `src/techniques/`: experimental border-artifact techniques
- `src/utils.py`: metrics, plots, and visualization utilities
- `outputs/`: saved evaluation results and figures

Swin results are stored in:

- `outputs/swin_none_best/`
- `outputs/swin_technique1_best/`
- `outputs/swin_technique2_best/`
- `outputs/swin_technique3_best/`

## Evaluation Outputs

Depending on the run, output folders contain:

- Per-class and macro metrics in `metrics.csv`
- Confusion matrices
- Precision-recall curves
- Training and validation plots
- Grad-CAM visualizations
- Proxy robustness comparison files

## Setup

Use Python 3.10 or later.

The supplied `requirements.txt` pins CUDA-specific PyTorch packages for a compatible NVIDIA environment. These dependencies are not a standard Mac installation configuration.

Install dependencies in a suitable Python environment:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python dataset_setup.py
```

The setup script downloads the dataset through KaggleHub when the expected dataset directory is absent. Internet access and any required Kaggle access are necessary.

Generated split CSV files are stored in:

```text
data/isic2019/splits/
```

## Training

Train standard Swin:

```bash
python train.py --model swin --technique none --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0
```

Train a technique variant:

```bash
python train.py --model swin --technique technique1 --epochs 50 --batch_size 64 --lr 1e-4 --device cuda:0
```

Replace `technique1` with `technique2` or `technique3` to run the other configurations.

Best checkpoints are saved under `models/`.

## Evaluation

Evaluate standard Swin:

```bash
python test.py --model swin --ckpt models/swin_none_best.pth --device cuda:0
```

Evaluate Technique 1:

```bash
python test.py --model swin --ckpt models/swin_technique1_best.pth --device cuda:0
```

For the other variants, use the corresponding checkpoint filename.

Dataset files and trained checkpoints are required for evaluation and are not included in this repository.

For CPU execution, replace `--device cuda:0` with `--device cpu`. A compatible dependency installation is still required, and training may be slow.

## Limitations

- The generated ruler/no-ruler proxy splits contain the same test images. Their comparison checks pipeline consistency rather than actual ruler-specific robustness.
- Improvements in classification metrics do not establish reduced ruler bias.
- Image-level splitting does not establish patient- or lesion-level separation.
- The results represent individual experiment runs without uncertainty estimates.
- Class imbalance and limited minority-class performance affect interpretation.
- This academic project is not a clinically validated diagnostic tool.

## Future Work

- Evaluate on genuinely annotated ruler-present and ruler-absent subsets.
- Use patient- or lesion-level splits where identifiers are available.
- Repeat experiments across multiple random seeds.
- Tune the strength of augmentation, attention regularization, and masking.
