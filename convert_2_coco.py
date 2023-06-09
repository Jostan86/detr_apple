
import json
from shapely.geometry import Polygon

files_base_directory = '/home/jostan/Documents/Amodal_Fruit_Sizing/datasets/data/gt_json/'

files_sub_directory = ['train/via_region_data_instance.json', 'train/via_region_data_amodal.json',
            'val/via_region_data_instance.json', 'val/via_region_data_amodal.json',
            'test/via_region_data_instance.json', 'test/via_region_data_amodal.json']
save_base_directory = '/home/jostan/Documents/detr/coco_apples/annotations_one_cat/'

save_sub_directory = ['modal/instances_train.json', 'amodal/instances_train.json',
                        'modal/instances_val.json', 'amodal/instances_val.json',
                        'modal/instances_test.json', 'amodal/instances_test.json']


# with open('/home/jostan/Documents/detr/coco2017/annotations/instances_val2017.json') as f:
#     data_coco = json.load(f)

for idx, (file_path, save_path) in enumerate(zip(files_sub_directory, save_sub_directory)):
    file_path = files_base_directory + file_path
    save_path = save_base_directory + save_path

    with open(file_path) as f:
        annotation_data = json.load(f)

    info = {
        'year': 2022,
        'version': '2.0',
        'description': 'Amodal Fruit Sizing Dataset',
        'contributor': 'Jordi Gené Mola; Mar Ferrer Ferrer; Eduard Gregorio López; Joan Ramon Rosell Polo; Veronica Vilaplana; Javier Ruiz Hidalgo; Josep Ramon Morros',
        'url': 'https://zenodo.org/record/7260694#.Y5Blu3bMJPY',
        'date_created': '2022/10/04'
    }

    licenses = [{
        'id': 1,
        'name': 'Creative Commons Attribution 4.0 International',
        'url': 'http://creativecommons.org/licenses/by/4.0/'
    }]

    # categories = data_coco['categories']
    categories = [{
        'id': 1,
        'name': 'apple',
        'supercategory': 'fruit',
    }]

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1
    for idx, kk in enumerate(annotation_data.keys()):
        v = annotation_data[kk]
        image_dict = {
            'id': image_id,
            'width': 1300,
            'height': 1300,
            'file_name': v['filename'],
            'license': 1,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': '2022/10/04'
        }
        images.append(image_dict)


        for region in v['regions']:
            all_points_x = v['regions'][region]['shape_attributes']['all_points_x']
            all_points_y = v['regions'][region]['shape_attributes']['all_points_y']

            min_x, min_y = min(all_points_x), min(all_points_y)
            max_x, max_y = max(all_points_x), max(all_points_y)
            width = max_x - min_x
            height = max_y - min_y

            polygon_coco = []
            polygon_area = []
            for x, y in zip(all_points_x, all_points_y):
                polygon_coco.append(x)
                polygon_coco.append(y)
                polygon_area.append((x, y))
            polygon_area = Polygon(polygon_area)
            area = polygon_area.area

            annotation_dict = {
                'id': annotation_id,
                'image_id': image_id,
                # 'category_id': 53,
                'category_id': 1,
                'segmentation': [polygon_coco],
                'area': area,
                'bbox': [min_x, min_y, width, height],
                'iscrowd': 0
            }
            annotations.append(annotation_dict)
            annotation_id += 1

        image_id += 1


    coco_data = {
        'info': info,
        'licenses': licenses,
        'categories': categories,
        'images': images,
        'annotations': annotations
    }

    with open(save_path, 'w') as f:
        json.dump(coco_data, f)