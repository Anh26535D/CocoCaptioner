import os
import numpy as np
import random

import torch
import ruamel.yaml as yaml
from transformers import CLIPModel

from dataset import create_dataset, create_loader, coco_collate_fn


def save_clip(clip_save_path):
    if not os.path.exists(clip_save_path):
        os.makedirs(clip_save_path)
        
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model_clip.save_pretrained(clip_save_path)
    print("CLIP model saved!")


def main(config):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds, val_ds, test_ds = create_dataset(config)
    train_loader = create_loader(train_ds, config['batch_size_train'], 0, True, coco_collate_fn)
    val_loader = create_loader(val_ds, config['batch_size_test'], 0, False, coco_collate_fn)
    test_loader = create_loader(test_ds, config['batch_size_test'], 0, False, coco_collate_fn)

    clip_path = './models/clip'
    if not os.path.exists(clip_path):
        save_clip(clip_path)
    model_clip = CLIPModel.from_pretrained(clip_path)

    for i, (image, caption, image_id, captions) in enumerate(train_loader):
        feature_images = model_clip.get_image_features(image)
        print(feature_images.shape)
        print(caption)
        print(image_id)
        print(captions)
        break


if __name__ == '__main__':
    yaml = yaml.YAML(typ='safe')
    config = yaml.load(open('./configs/coco_config.yaml', 'r'))
    main(config)
