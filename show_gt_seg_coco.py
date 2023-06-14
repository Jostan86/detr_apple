# import os
# from pycocotools.coco import COCO
# from pycocotools import mask
# from skimage.io import imread
# from skimage.measure import find_contours
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.path import Path
#
# dataDir = '/home/jostan/Documents/detr/coco_apples/'  # Replace with the path to your dataset
# dataType = 'train_apples'  # Replace with the appropriate dataset split
# # annFile = 'new_dataset.json'
# annotation_file_path2 = './coco_apples/annotations_one_cat/modal/instances_train.json'
# coco = COCO(annotation_file_path2)  # Path to the COCO annotations file

# coco = COCO(annFile)

# imgIds = coco.getImgIds()  # Get all image IDs
# imgId = imgIds[20]  # Select the first image
# img = coco.loadImgs(imgId)[0]  # Load the image information
# imgPath = os.path.join(dataDir, dataType, img['file_name'])  # Path to the image file
# print("Image path: ", imgPath)
# image = imread(imgPath)
# plt.imshow(image)
#
# annIds = coco.getAnnIds(imgIds=img['id'])  # Get annotation IDs for the image
# annotations = coco.loadAnns(annIds)  # Load the annotations
#
# for annotation in annotations:
#     segmentation = annotation['segmentation']
#     category_id = annotation['category_id']
#     category_info = coco.loadCats(category_id)[0]
#     color = [1,1,1]
#
#     polygons = [segmentation]
#
#     # Plot each polygon
#     for polygon in polygons:
#         vertices = []
#         segment = polygon[0]
#         xs = segment[0::2]
#         ys = segment[1::2]
#         for (x, y) in zip(xs, ys):
#             vertices.append((x, y))
#
#         path = Path(vertices)
#         patch = patches.PathPatch(path, facecolor=color, edgecolor='none', alpha=0.7)
#         plt.gca().add_patch(patch)
#
#     # Add a bounding box around the segmentation
#     bbox = annotation['bbox']
#     rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
#                              edgecolor=color, facecolor='none')
#     plt.gca().add_patch(rect)
#
# plt.axis('off')
# plt.show()


import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

dataDir = '/home/jostan/Documents/detr/coco_apples/'  # Replace with the path to your dataset
dataType = 'train_apples'  # Replace with the appropriate dataset split
annFile = './coco_apples/annotations_one_cat/modal/instances_train.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

nms=[cat['name'] for cat in cats]
print('Categories: {}'.format(nms))

nms = set([cat['supercategory'] for cat in cats])
print('Super-categories: {}'.format(nms))

# load and display image
catIds = coco.getCatIds(catNms=['apple'])
imgIds = coco.getImgIds(catIds=catIds )

img_id = imgIds[np.random.randint(0,len(imgIds))]
print('Image n°{}'.format(img_id))

img = coco.loadImgs(img_id)[0]

img_name = '%s/%s/%s'%(dataDir, dataType, img['file_name'])
print('Image name: {}'.format(img_name))

I = io.imread(img_name)
plt.figure()


annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
anns = coco.loadAnns(annIds)

# load and display instance annotations
plt.imshow(I)
coco.showAnns(anns, draw_bbox=False)
coco.showAnns(anns, draw_bbox=True)
plt.show()