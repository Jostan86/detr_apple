# Shows the segmentations for the apple images
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import json
from PIL import Image
import os
import pprint

import numpy as np
import cv2
from shapely.geometry import Polygon


def create_polygons(x_values, y_values):
    img_width = 1300
    img_height = 1300
    # Create a binary mask image from the coordinates
    mask = np.zeros((img_width, img_height), dtype=np.uint8)
    mask[y_values, x_values] = 255

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to polygons
    polygons = []
    for contour in contours:
        contour = np.squeeze(contour)
        polygons.append(Polygon(contour))

    return polygons

def display_segmentation(file_name, show_bbox=True, instance=True):

    if instance:
        # Load JSON file
        with open('/home/jostan/Documents/Amodal_Fruit_Sizing/datasets/data/gt_json/test/via_region_data_instance'
                  '.json') \
                as f:
            data = json.load(f)
    else:
        # Load JSON file
        with open('/home/jostan/Documents/Amodal_Fruit_Sizing/datasets/data/gt_json/test/via_region_data_amodal.json') \
                as f:
            data = json.load(f)
    # pprint.pprint(data['_MG_2644_22.png1487915'])#['regions']['2']['shape_attributes'])
    keys = data.keys()
    # # View the contents of the JSON data
    for i, key in enumerate(keys):
        if data[key]['filename'] == file_name:
            print('Found the file')
            print(key)
            key_save = key

    data = data[key_save]['regions']

    # Load image
    image_path = '/home/jostan/Documents/Amodal_Fruit_Sizing/datasets/data/images/test/' + file_name
    image = Image.open(image_path)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    i = 0
    to_use = 0
    # Plot segmentations
    for region in data:
        i += 0.7

        active = 10
        if to_use < active:
            to_use += 1
            continue
        if to_use == active+1:
            break
        to_use += 1

        # Get segmentation coordinates
        region = data[region]
        shape_attributes = region['shape_attributes']
        if shape_attributes['name'] == 'polygon':
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']

            # polygons = create_polygons(all_points_x, all_points_y)
            polygon_area = []
            for x, y in zip(all_points_x, all_points_y):
                polygon_area.append((x, y))
            polygon_area = Polygon(polygon_area)
            area = polygon_area.area
            print(area)
            polygon = patches.Polygon(list(zip(all_points_x, all_points_y)), closed=True)

            if show_bbox:
                # Calculate bounding box
                min_x, min_y = min(all_points_x), min(all_points_y)
                max_x, max_y = max(all_points_x), max(all_points_y)
                width = max_x - min_x
                height = max_y - min_y

                # Create bounding box patch
                bounding_box = patches.Rectangle((min_x, min_y), width, height, linewidth=2, edgecolor='r', facecolor='none')

            # Assign a different color to each polygon
            color = cm.rainbow(i / len(data))
            polygon.set_facecolor(color)

            # Add polygon to the axes
            ax.add_patch(polygon)

            if show_bbox:
                ax.add_patch(bounding_box)

    # Hide axes
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Set figure size
    fig.set_size_inches(12, 12)

    # Show the plot
    plt.show()

img_dir = '/home/jostan/Documents/Amodal_Fruit_Sizing/datasets/data/images/test/'
img_file_names = []

# Get all the file names in the directory
for file in os.listdir(img_dir):
    if file.endswith(".png"):
        img_file_names.append(file)

# Arrange the files alphabetically
img_file_names.sort()

display_segmentation(img_file_names[2], show_bbox=True, instance=True)