# CR-ReID

Official implementation for **CR-ReID**, a cloth-changing person re-identification framework based on EVA02 visual features, semantic pedestrian attributes, and clothing-region preprocessing.

The code supports four cloth-changing ReID benchmarks:

- [PRCC](http://www.isee-ai.cn/%7Eyangqize/clothing.html)
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
- [Celeb-reID-light](https://github.com/Huang-3/Celeb-reID)
- [LaST](https://github.com/shuxjweb/last.git)

## Highlights

- EVA02-based visual backbone.
- Attribute-guided ReID with 105-dimensional pedestrian semantic attributes.
- Clothing-attribute masking for cloth-changing robustness.
- RACS-style clothing region preprocessing for dual-stream training.
- Training and evaluation scripts for PRCC, LTCC, Celeb-reID-light, and LaST.

## Project Structure

```text
CR-ReID/
|-- config/                 # Default configuration
|-- configs/                # Dataset-specific yaml configs
|   |-- prcc/
|   |-- ltcc/
|   |-- Celeb_light/
|   `-- last/
|-- data/                   # Dataset definitions, dataloaders, samplers, transforms
|-- loss/                   # ReID losses
|-- model/                  # EVA02 backbone and meta-attribute modules
|-- processor/              # Training and inference loops
|-- racs/                   # Clothing-region preprocessing scripts
|-- solver/                 # Optimizer and scheduler
|-- tools/                  # Evaluation utilities
|-- utils/                  # Logger, metrics, reranking, IO helpers
|-- train.py                # Training entry
|-- test.py                 # Evaluation entry
|-- requirement.txt         # Python dependencies
`-- readme.md
```

## Key Environment and Library Versions

The main environment used in this project is:

| Package       | Version        |
|---------------|----------------|
| Python        | 3.12           |
| CUDA          | 12.1           |
| PyTorch       | 2.5.1+cu121    |
| torchvision   | 0.20.1+cu121   |
| torchaudio    | 2.5.1+cu121    |
| yacs          | 0.1.8          |
| numpy         | 2.0.2          |
| tensorflow    | 2.18.0         |
| scikit-learn  | 1.6.1          |
| opencv-python | 4.11.0.86      |
| timm          | 1.0.15         |
| torchreid     | 0.2.5          |

Install dependencies:

```bash
pip install -r requirement.txt
```

If you install PyTorch manually, make sure the PyTorch, torchvision, torchaudio, and CUDA versions are compatible with your GPU driver.

## Prepare Datasets

Download the cloth-changing person ReID datasets:

- [PRCC](http://www.isee-ai.cn/%7Eyangqize/clothing.html)
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
- [Celeb-reID-light](https://github.com/Huang-3/Celeb-reID)
- [LaST](https://github.com/shuxjweb/last.git)

Place the datasets under one root directory, and set `DATA.ROOT` in the config file or command line. The dataset loaders expect the following structure:

```text
DATA_ROOT/
|-- prcc/
|   |-- rgb/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- PAR_PETA_105.txt
|-- LTCC/
|   |-- train/
|   |-- query/
|   |-- test/
|   `-- PAR_PETA_105.txt
|-- Celeb-reID-light/
|   |-- train/
|   |-- query/
|   |-- gallery/
|   `-- PAR_PETA_105.txt
`-- last/
    |-- train/
    |-- val/
    |   |-- query/
    |   `-- gallery/
    |-- test/
    |   |-- query/
    |   `-- gallery/
    `-- PAR_PETA_105.txt
```



## Preprocessing for Dual-Stream Learning

The training script runs clothing-region preprocessing before training. The preprocessing reads original images and clothing masks, then produces processed images where clothing regions are adaptively blurred/desaturated. Original images with the `ori` suffix are also copied into the output folder for dual-stream learning.

Related config fields:

```yaml
PREPROCESS:
  ENABLE: True
  INPUT_FOLDER: "/path/to/input/images"
  MASK_FOLDER: "/path/to/clothing/masks"
  OUTPUT_FOLDER: "/path/to/output/images"
  ORI_FOLDER: "/path/to/original/images"
```

Folder meanings:

- `INPUT_FOLDER`: images to be processed.
- `MASK_FOLDER`: clothing-region masks corresponding to the input images.
- `OUTPUT_FOLDER`: destination folder for processed images.
- `ORI_FOLDER`: original images copied into `OUTPUT_FOLDER` with the `ori` suffix.

Notes:

- Image filenames in `INPUT_FOLDER` and `MASK_FOLDER` should correspond exactly.
- `OUTPUT_FOLDER` is created automatically if it does not exist.
- PRCC uses `racs/prcc_racs.py`; the other datasets use `racs/ltcc_racs.py`.
- In the current `train.py`, preprocessing is called directly before dataloader construction. If your processed images already exist, you can reuse them by setting the folders correctly or by disabling/commenting out the preprocessing call in `train.py`.

Example dual-stream training folder:

```text
train/
|-- xxx.jpg       # processed image
`-- xxxori.jpg    # original image
```

## Pretrained Models

The pretrained models trained on PRCC, LTCC, Celeb-reID-light, and LaST are available on [Baidu Netdisk](https://pan.baidu.com/s/1YEgdxaWGdOdVGmypMp7TFg) with the extraction code: CCRD.
After downloading a checkpoint, set it with `TEST.WEIGHT` when running evaluation.



## Evaluation

Evaluate a trained model by setting `TEST.WEIGHT`:

```bash
python test.py --config_file configs/prcc/eva02_l_maskmeta_random.yml DATA.ROOT /path/to/DATA_ROOT TEST.WEIGHT /path/to/eva02_l_meta_best.pth
```

For PRCC, the evaluation reports both:

- Clothes-changing setting.
- Standard setting.

For LTCC, the evaluation reports:

- Clothes-changing setting.
- Standard setting.

For Celeb-reID-light and LaST, the evaluation reports Rank-1, Rank-5, Rank-10, and mAP.

## TensorBoard

Training writes TensorBoard logs under `OUTPUT_DIR`:

```text
OUTPUT_DIR/
|-- train/
|-- rank/
`-- mAP/
```

Launch TensorBoard:

```bash
tensorboard --logdir OUTPUT_DIR
```

## Acknowledgements

This project uses or refers to the following excellent works:

- [EVA / EVA02](https://github.com/baaivision/EVA)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [MADE](https://github.com/moon-wh/MADE)
- PRCC, LTCC, Celeb-reID-light, and LaST dataset projects

## Citation

If this code is useful for your research, please cite this project and the corresponding dataset papers.
