import torch
from models import build_model
from main2 import get_args_parser
import argparse
import numpy as np
import torchvision.transforms as T
from PIL import Image
torch.set_grad_enabled(False)
from detectron2.structures import Instances

class ApplePredictor:
    def __init__(self, coco_path, modal_seg_path, amodal_seg_path, synth_included, backbone, dc5):
        # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])



        # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        # args = parser.parse_args()
        args2 = argparse.Namespace()
        args2.coco_path = coco_path
        args2.masks = True
        args2.epochs = 25
        args2.lr_drop = 15


        args2.lr = 1e-4
        args2.lr_backbone = 1e-5
        args2.batch_size = 2
        args2.weight_decay = 1e-4
        args2.clip_max_norm = 0.1
        args2.backbone = backbone
        args2.dilation = dc5
        args2.position_embedding = 'sine'
        args2.enc_layers = 6
        args2.dec_layers = 6
        args2.dim_feedforward = 2048
        args2.hidden_dim = 256
        args2.dropout = 0.1
        args2.nheads = 8
        args2.num_queries = 100
        args2.pre_norm = False
        args2.aux_loss = True
        args2.set_cost_class = 1.0
        args2.set_cost_bbox = 5.0
        args2.set_cost_giou = 2.0
        args2.mask_loss_coef = 1.0
        args2.dice_loss_coef = 1.0
        args2.bbox_loss_coef = 5.0
        args2.giou_loss_coef = 2.0
        args2.eos_coef = 0.1
        args2.num_classes = None
        args2.output_dir = ''
        args2.device = 'cuda'
        args2.seed = 42
        args2.resume = ''
        args2.start_epoch = 0
        args2.eval = False
        args2.num_workers = 2
        args2.world_size = 1
        args2.dist_url = 'env://'




        # fix the seed for reproducibility
        # seed = args.seed + utils.get_rank()
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        args2.frozen_weights = modal_seg_path
        if synth_included:
            args2.dataset_file = 'coco_apples_modal_synth'
        else:
            args2.dataset_file = 'coco_apples_modal'

        self.model_modal, criterion, self.postprocessors_modal = build_model(args2)
        self.model_modal.to('cpu')

        args2.frozen_weights = amodal_seg_path
        if synth_included:
            args2.dataset_file = 'coco_apples_amodal_synth'
        else:
            args2.dataset_file = 'coco_apples_amodal'

        self.model_amodal, criterion, self.postprocessors_amodal = build_model(args2)

        self.bbox_postprocessor_modal = self.postprocessors_modal['bbox']
        self.segmentation_postprocessor_modal = self.postprocessors_modal['segm']
        self.bbox_postprocessor_amodal = self.postprocessors_amodal['bbox']
        self.segmentation_postprocessor_amodal = self.postprocessors_amodal['segm']

        checkpoint_modal = torch.load('/home/jostan/Documents/detr/logdirs/modalSeg/checkpoint.pth', map_location='cpu')
        checkpoint_amodal = torch.load('/home/jostan/Documents/detr/logdirs/amodalSeg/checkpoint.pth', map_location='cpu')

        self.model_modal.load_state_dict(checkpoint_modal['model'], strict=False)
        self.model_amodal.load_state_dict(checkpoint_amodal['model'], strict=False)

        self.model_modal.eval()
        self.model_amodal.eval()



    def match_masks(self, masks_a: np.ndarray, masks_m: np.ndarray, threshold=100):
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

    def numpyize_results(self, results):
        results['scores'] = results['scores'].numpy()
        results['boxes'] = results['boxes'].numpy()
        results['labels'] = np.zeros(len(results['labels']), dtype=int)
        results['masks'] = results['masks'].squeeze(1).numpy()

        return results

    def sort_results(self, results, indices):
        results['scores'] = results['scores'][indices]
        results['boxes'] = results['boxes'][indices]
        results['labels'] = results['labels'][indices]
        results['masks'] = results['masks'][indices]

        return results

    def make_prediction(self, img_filepath, score_threshold=0.6, match_dist_threshold=100):

        im = Image.open(img_filepath)
        img = self.transform(im).unsqueeze(0)
        out_modal = self.model_modal(img)
        out_amodal = self.model_amodal(img)

        batch_size = 1
        height = 1300
        width = 1300

        target_sizes = torch.stack([torch.tensor([height, width])] * batch_size)

        results_m = self.bbox_postprocessor_modal(out_modal, target_sizes)
        results_m = self.segmentation_postprocessor_modal(results_m, out_modal, target_sizes, target_sizes)[0]
        results_a = self.bbox_postprocessor_amodal(out_amodal, target_sizes)
        results_a = self.segmentation_postprocessor_amodal(results_a, out_amodal, target_sizes, target_sizes)[0]

        results_m = self.numpyize_results(results_m)
        results_a = self.numpyize_results(results_a)

        keep_m = results_m['scores'] > score_threshold
        keep_a = results_a['scores'] > score_threshold

        results_m = self.sort_results(results_m, keep_m)
        results_a = self.sort_results(results_a, keep_a)

        matches_a, matches_m = self.match_masks(results_a['masks'], results_m['masks'], threshold=match_dist_threshold)

        results_m = self.sort_results(results_m, matches_m)
        results_a = self.sort_results(results_a, matches_a)

        scores_mean = (results_a['scores'] + results_m['scores']) / 2

        instances = Instances((height, width), pred_boxes=results_a['boxes'], scores=scores_mean,
                                pred_classes=results_a['labels'], pred_masks=results_a['masks'],
                                pred_visible_masks=results_m['masks'])

        return instances






