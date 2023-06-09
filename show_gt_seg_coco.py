import os
from pycocotools.coco import COCO
from pycocotools import mask
from skimage.io import imread
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

dataDir = '/home/jostan/Documents/detr/coco_apples/'  # Replace with the path to your dataset
dataType = 'train2017'  # Replace with the appropriate dataset split
# annFile = 'new_dataset.json'
annotation_file_path2 = 'coco2017/annotations/instances_train2017.json'
coco = COCO(annotation_file_path2)  # Path to the COCO annotations file

# coco = COCO(annFile)

imgIds = coco.getImgIds()  # Get all image IDs
imgId = imgIds[20]  # Select the first image
img = coco.loadImgs(imgId)[0]  # Load the image information
imgPath = os.path.join(dataDir, dataType, img['file_name'])  # Path to the image file
print("Image path: ", imgPath)
image = imread(imgPath)
plt.imshow(image)

annIds = coco.getAnnIds(imgIds=img['id'])  # Get annotation IDs for the image
annotations = coco.loadAnns(annIds)  # Load the annotations

for annotation in annotations:
    segmentation = annotation['segmentation']
    category_id = annotation['category_id']
    category_info = coco.loadCats(category_id)[0]
    color = [1,1,1]

    polygons = [segmentation]

    # Plot each polygon
    for polygon in polygons:
        vertices = []
        segment = polygon[0]
        xs = segment[0::2]
        ys = segment[1::2]
        for (x, y) in zip(xs, ys):
            vertices.append((x, y))

        path = Path(vertices)
        patch = patches.PathPatch(path, facecolor=color, edgecolor='none', alpha=0.7)
        plt.gca().add_patch(patch)

    # Add a bounding box around the segmentation
    bbox = annotation['bbox']
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                             edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)

plt.axis('off')
plt.show()



