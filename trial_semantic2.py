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
args.frozen_weights = '/home/jostan/Documents/detr/logdirs/modalSeg/checkpoint.pth'
# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model_modal, criterion, postprocessors_modal = build_model(args)
model_modal.to('cpu')

args.frozen_weights = '/home/jostan/Documents/detr/logdirs/amodalSeg/checkpoint.pth'
model_amodal, criterion, postprocessors_amodal = build_model(args)

bbox_postprocessor_modal = postprocessors_modal['bbox']
segmentation_postprocessor_modal = postprocessors_modal['segm']
bbox_postprocessor_amodal = postprocessors_amodal['bbox']
segmentation_postprocessor_amodal = postprocessors_amodal['segm']

checkpoint_modal = torch.load('logdirs/modalSeg/checkpoint.pth', map_location='cpu')
checkpoint_amodal = torch.load('logdirs/amodalSeg/checkpoint.pth', map_location='cpu')

model_modal.load_state_dict(checkpoint_modal['model'], strict=False)
model_amodal.load_state_dict(checkpoint_amodal['model'], strict=False)

model_modal.eval()
model_amodal.eval()

img_dir = './coco_apples/test_apples/'
save_dir = 'apple_images_output_seg2/'
img_file_names = []

# Get all the file names in the directory
for file in os.listdir(img_dir):
    if file.endswith(".png"):
        img_file_names.append(file)

# Arrange the files alphabetically
img_file_names.sort()


def match_masks(masks_a: np.ndarray, masks_m: np.ndarray, threshold=100):
    # This will be a fun little function where I find the middle of all the amodal and modal masks then try to match
    # the modal masks to the amodal masks based on distance. I'll have to first match the two closest centers,
    # then remove those from the list and repeat until there are no more of one of the sets of masks. If at any point the
    # distance between the two closest centers is greater than a threshold, then that will be all the matches.
    # I'll return 2 numpy arrays, with the masks arranges in the correct orders.

    # Get the centers of the masks
    centers_a = []
    centers_m = []
    for mask in masks_a:
        centers_a.append(np.mean(np.where(mask), axis=1))
    for mask in masks_m:
        centers_m.append(np.mean(np.where(mask), axis=1))

    # Initialize the matches list
    matches_a = []
    matches_m = []

    idx_a_og = [i for i in range(len(centers_a))]
    idx_m_og = [i for i in range(len(centers_m))]

    # Loop until one of the sets of masks is empty
    while len(centers_a) > 0 and len(centers_m) > 0:
        # Find the closest centers
        closest_dist = 100000
        closest_a_idx = None
        closest_m_idx = None

        for idx_a, center_a in enumerate(centers_a):
            for idx_m, center_m in enumerate(centers_m):
                dist = np.linalg.norm(center_a - center_m)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_a_idx = idx_a
                    closest_m_idx = idx_m

        # If the closest distance is greater than the threshold, then we're done
        if closest_dist > threshold:
            break

        # Add the match to the list
        matches_a.append(idx_a_og.pop(closest_a_idx))
        matches_m.append(idx_m_og.pop(closest_m_idx))

        # Remove the closest centers from the list
        centers_a.pop(closest_a_idx)
        centers_m.pop(closest_m_idx)

    return matches_a, matches_m

def numpyize_results(results):
    results['scores'] = results['scores'].numpy()
    results['boxes'] = results['boxes'].numpy()
    results['labels'] = np.zeros(len(results['labels']), dtype=int)
    results['masks'] = results['masks'].squeeze(1).numpy()

    return results

def sort_results(results, indices):
    results['scores'] = results['scores'][indices]
    results['boxes'] = results['boxes'][indices]
    results['labels'] = results['labels'][indices]
    results['masks'] = results['masks'][indices]

    return results


img_count = 0
for file_name in img_file_names[4:]:
    if img_count == 100:
        break
    img_count += 1

    im = Image.open(img_dir + file_name)
    img = transform(im).unsqueeze(0)
    out_modal = model_modal(img)
    out_amodal = model_amodal(img)

    batch_size = 1
    height = 1300
    width = 1300

    target_sizes = torch.stack([torch.tensor([height, width])] * batch_size)

    results_m = bbox_postprocessor_modal(out_modal, target_sizes)
    results_m = segmentation_postprocessor_modal(results_m, out_modal, target_sizes, target_sizes)[0]
    results_a = bbox_postprocessor_amodal(out_amodal, target_sizes)
    results_a = segmentation_postprocessor_amodal(results_a, out_amodal, target_sizes, target_sizes)[0]

    results_m = numpyize_results(results_m)
    results_a = numpyize_results(results_a)

    keep_m = results_m['scores'] > 0.7
    keep_a = results_a['scores'] > 0.7

    results_m = sort_results(results_m, keep_m)
    results_a = sort_results(results_a, keep_a)

    matches_a, matches_m = match_masks(results_a['masks'], results_m['masks'])

    results_m = sort_results(results_m, matches_m)
    results_a = sort_results(results_a, matches_a)

    # instances = Instances((height, width), pred_boxes=results_a['boxes'], scores=results_a['scores'],
    #                         pred_classes=results_a['labels'], pred_masks=results_a['masks'],
    #                         pred_visible_masks=results_m['masks'])

    instances_m = Instances((height, width), pred_boxes=results_m['boxes'], scores=results_m['scores'],
                          pred_classes=results_m['labels'], pred_masks=results_m['masks'])

    x = (results_a['scores'] + results_m['scores']) / 2

    instances_a = Instances((height, width), pred_boxes=results_a['boxes'], scores=results_a['scores'],
                            pred_classes=results_a['labels'], pred_masks=results_a['masks'])
    #
    # results_combined = {
    #     'scores': results_m['scores'],
    #     'boxes': results_m['boxes'],
    #     'labels': results_m['labels'],
    #     'masks': results_m['masks'],
    #     'visible_masks': results_a['masks']
    # }


    # metadata =     meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    metadata = MetadataCatalog.get("coco_2017_val")


    # Create a Visualizer object with the original image and metadata
    v_m = Visualizer(im, scale=0.7)
    v_a = Visualizer(im, scale=0.7)

    # Draw the instance predictions
    out_modal = v_m.draw_instance_predictions(predictions=instances_m)
    out_amodal = v_a.draw_instance_predictions(predictions=instances_a)

    # Get the visualized image
    vis_image_m = out_modal.get_image()
    vis_image_a = out_amodal.get_image()

    cv2.imshow('yo', out_modal.get_image()[:, :, ::-1])
    cv2.imshow('yo2', out_amodal.get_image()[:, :, ::-1])

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite(save_dir + 'modal' + str(img_count) + file_name , out.get_image()[:, :, ::-1])






