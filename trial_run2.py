import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

# import ipywidgets as widgets
# from IPython.display import display, clear_output
import os
import torch
import torch, torchvision
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs,
                               threshold=0.7):
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return probas_to_keep, bboxes_scaled

def plot_finetuned_results(pil_img, prob=None, boxes=None, save=False, save_path=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          # text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          text = 'apple'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if save:
        plt.savefig(save_dir + img_file_name)
    # Close the plot
        plt.close()
    else:
        plt.show()


def run_worflow(my_image, my_model, save_path=None, save=False):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    outputs = my_model(img)

    for threshold in [0.9]:
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs,
                                                                   threshold=threshold)

        plot_finetuned_results(my_image,
                               probas_to_keep,
                               bboxes_scaled,
                               save=save,
                               save_path=save_path,)


model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=2)

checkpoint = torch.load('logdirs/11/checkpoint.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval()


img_dir = '/home/jostan/Documents/detr/coco_apples/test_apples/'
save_dir = 'apple_images_output/'
img_file_names = []
# Get all the file names in the directory
for file in os.listdir(img_dir):
    if file.endswith(".png"):
        img_file_names.append(file)
#
#
img_count = 0
for img_file_name in img_file_names:
    if img_count == 40:
        break
    img_count += 1
    im = Image.open(img_dir + img_file_name)

    run_worflow(im, model, save=True, save_path=save_dir + img_file_name)
    print("Done with image: " + str(img_count) + " of " + str(len(img_file_names)) + " images.")