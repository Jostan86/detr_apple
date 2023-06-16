# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .coco import build_apple_amodal_dataset, build_apple_modal_dataset, build_apple_amodal_synth_dataset, \
    build_apple_modal_synth_dataset, build_apple_dataset_both_masks


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':

        return build_coco(image_set, args)
    if args.dataset_file == 'coco_apples_modal':
        return build_apple_modal_dataset(image_set, args)
    if args.dataset_file == 'coco_apples_amodal':
        return build_apple_amodal_dataset(image_set, args)
    if args.dataset_file == 'coco_apples_amodal_synth':
        return build_apple_amodal_synth_dataset(image_set, args)
    if args.dataset_file == 'coco_apples_modal_synth':
        return build_apple_modal_synth_dataset(image_set, args)
    if args.dataset_file == 'coco_apples_both_masks':
        return build_apple_dataset_both_masks(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
