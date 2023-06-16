
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils

import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torchvision
import pylab
# pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from datasets import build_dataset, get_coco_api_from_dataset

import datasets.transforms as T

# dataDir = '/home/jostan/Documents/detr/coco_apples/'  # Replace with the path to your dataset
# dataType = 'train_apples'  # Replace with the appropriate dataset split
# annFile = './coco_apples/annotations_one_cat/modal/instances_train.json'


# class CocoDetection(torchvision.datasets.CocoDetection):
class CocoDetectionSplitter(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetectionSplitter, self).__init__(img_folder, ann_file)

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetectionSplitter, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self, img_folder, ann_file_modal, ann_file_amodal, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file_modal)
        self.amodal_loader = CocoDetectionSplitter(img_folder, ann_file_amodal, transforms, return_masks)
        self.modal_loader = CocoDetectionSplitter(img_folder, ann_file_modal, transforms, return_masks)

    def __getitem__(self, idx):
        _, target_modal = self.modal_loader[idx]
        img, target = self.amodal_loader[idx]

        target['amodal_masks'] = target.pop('masks')
        target['modal_masks'] = target_modal['masks']

        return img, target



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

dataDir = '/home/jostan/Documents/detr/coco_apples/train_apples/annotations_one_cat/modal/'  # Replace with the path to
# your dataset
dataType = 'train_apples'  # Replace with the appropriate dataset split
# annFile = '/home/jostan/Documents/detr/coco_apples_synth/annotations/amodal/instances_train.json'

img_folder = '/home/jostan/Documents/detr/coco_apples/train_apples/'
annFile_modal = '/home/jostan/Documents/detr/coco_apples/annotations_one_cat/modal/instances_train.json'
annFile_amodal = '/home/jostan/Documents/detr/coco_apples/annotations_one_cat/amodal/instances_train.json'

image_set = 'train'
test_coco = CocoDetection(img_folder, annFile_modal, annFile_amodal, transforms=make_coco_transforms(image_set),
                          return_masks=True)

# x = test_coco[4]
# x = test_coco[9]
# x = test_coco[6]
# x = test_coco[3]
sampler_train = torch.utils.data.RandomSampler(test_coco)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 1, drop_last=True)

data_loader_train = DataLoader(test_coco, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=2)

base_ds = get_coco_api_from_dataset(test_coco)
print(base_ds)
# for samples, targets in data_loader_train:
#     print(samples)
#     print(targets)
#     break
# # initialize COCO api for instance annotations
# coco=COCO(annFile)
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
#
# # nms=[cat['name'] for cat in cats]
# # print('Categories: {}'.format(nms))
# #
# # nms = set([cat['supercategory'] for cat in cats])
# # print('Super-categories: {}'.format(nms))
#
# # load and display image
# catIds = coco.getCatIds(catNms=['apple'])
# imgIds = coco.getImgIds(catIds=catIds )
#
# img_id = imgIds[np.random.randint(0,len(imgIds))]
# print('Image n°{}'.format(img_id))
#
# img = coco.loadImgs(img_id)[0]
#
# img_name = '%s/%s/%s'%(dataDir, dataType, img['file_name'])
# print('Image name: {}'.format(img_name))
#
# I = io.imread(img_name)
# plt.figure()
#
#
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
# anns = coco.loadAnns(annIds)
#
# # load and display instance annotations
# plt.imshow(I)
# coco.showAnns(anns, draw_bbox=False)
# coco.showAnns(anns, draw_bbox=True)
# plt.show()