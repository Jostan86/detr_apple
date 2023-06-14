import torch
from models import build_model
from main2 import get_args_parser
import argparse
from pathlib import Path
import util.misc as utils
import numpy as np
import random
import torchvision.transforms as T
import os
from PIL import Image
torch.set_grad_enabled(False)
import cv2
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
args.coco_path = '/home/jostan/Documents/detr/coco_apples'
args.masks = True
args.epochs = 25
args.lr_drop = 15
args.dataset_file = 'coco_apples_modal'
args.frozen_weights = '/home/jostan/Documents/detr/logdirs/10/checkpoint.pth'
# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model, criterion, postprocessors = build_model(args)
model.to('cpu')

# CLASSES = [
#     'apple'
# ]
#
# # Detectron2 uses a different numbering scheme, we build a conversion table
# coco2d2 = {}
# count = 0
# for i, c in enumerate(CLASSES):
#   if c != "N/A":
#     coco2d2[i] = count
#     count+=1
#
# # standard PyTorch mean-std input image normalization
# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#
# model, postprocessor = torch.hub.load('facebookresearch/detr',
#                        'detr_resnet101_instance',
#                        pretrained=False,
#                        num_classes=2,
#                        return_postprocessor=True)
#
bbox_postprocessor = postprocessors['bbox']
segmentation_postprocessor = postprocessors['segm']

checkpoint = torch.load('logdirs/10/checkpoint.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval()

img_dir = './coco_apples/test_apples/'
save_dir = 'apple_images_output_seg2/'
img_file_names = []

# Get all the file names in the directory
for file in os.listdir(img_dir):
    if file.endswith(".png"):
        img_file_names.append(file)

# Arrange the files alphabetically
img_file_names.sort()



img_count = 0
for file_name in img_file_names[4:]:
    if img_count == 100:
        break
    img_count += 1

    im = Image.open(img_dir + file_name)
    img = transform(im).unsqueeze(0)
    out = model(img)

    # # compute the scores, excluding the "no-object" class (the last one)
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]

    # threshold the confidence
    keep = scores > 0.85

    batch_size = 1
    height = 1300
    width = 1300
    out_logits, out_bbox = out['pred_logits'], out['pred_boxes']

    target_sizes = torch.stack([torch.tensor([height, width])] * batch_size)

    results = bbox_postprocessor(out, target_sizes)
    results = segmentation_postprocessor(results, out, target_sizes, target_sizes)

    # Filter out the predictions with low confidence scores
    scores = results[0]['scores']
    keep = scores > 0.85
    results[0]['scores'] = scores[keep].numpy()
    results[0]['boxes'] = results[0]['boxes'][keep].numpy()
    results[0]['labels'] = results[0]['labels'][keep].numpy()
    results[0]['masks'] = results[0]['masks'][keep].squeeze(1).numpy()
    results = results[0]
    instances = Instances((height, width), pred_boxes=results['boxes'], scores=results['scores'], pred_classes=results['labels'], pred_masks=results['masks'])

    # metadata =     meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    metadata = MetadataCatalog.get("coco_2017_val")


    # Create a Visualizer object with the original image and metadata
    v = Visualizer(im)

    # Draw the instance predictions
    out = v.draw_instance_predictions(predictions=instances)

    # Get the visualized image
    vis_image = out.get_image()

    cv2.imshow('yo', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(save_dir + '3_' + str(img_count) + file_name , out.get_image()[:, :, ::-1])





