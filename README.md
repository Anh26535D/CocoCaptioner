# Coco Image Captioning Project

## I. Prerequisites
1. Python 3.9 or higher
2. Coco Captions annotation files can be downloaded [here](https://cocodataset.org/#download)

## II. Installation
1. Create folder `data/coco_captions/annotations`, `img_root/coco_2014`, `output/models` and `output/results`
2. Extract Coco Captions annotation files to `data/coco_captions.annotations`. See `split_train_test.py` to find file format annotation that we convert to. Change config in `configs` folder.

## III. Usage
1. Create venv and install packages
```
    python3 -m venv .venv
```
```
    .\.venv\Scripts\activate
    pip install -r requirements.txt
```

2. Train model
```
    python coco_caption_main.py
```