import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from configs.default import get_cfg
# from configs.things_eval import get_cfg as get_things_cfg
# from configs.small_things_eval import get_cfg as get_small_things_cfg
from configs.sintel_inr import get_cfg
from core.utils.misc import process_cfg
import datasets_inr
from utils import flow_viz
from utils import frame_utils
from torchvision.utils import save_image

# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from raft import RAFT

from utils.utils import InputPadder, forward_interpolate, compute_out_of_boundary_mask



@torch.no_grad()
def validate_sintel_inr(model, image_root, flow_root,occlu_root=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets_inr.MpiSintel(split='training', dstype=dstype, image_root=image_root, flow_root=flow_root,occlu_root=occlu_root)
        epe_list = []
        if occlu_root:
            matched_epe_list = []
            unmatched_epe_list = []

        for val_id in tqdm(range(len(val_dataset)),desc=f'Validating Sintel {dstype}'):
            if occlu_root:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(flow_gt.unsqueeze(0)).squeeze(0)  # [H, W]
            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre = model(image1, image2)

            flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

            # save flow_pre
            flow_pre_png = flow_viz.flow_to_image(flow_pre.permute(1, 2, 0).numpy())
            flow_gt_png = flow_viz.flow_to_image(flow_gt.permute(1, 2, 0).numpy())
            # save_image(torch.from_numpy(flow_pre_png).permute(2, 0, 1)/255, f'logs/test/{dstype}/flow_pre_{val_id}.png')
            save_image([torch.from_numpy(flow_pre_png).permute(2, 0, 1)/255,torch.from_numpy(flow_gt_png).permute(2, 0, 1)/255], f'logs/test/{dstype}/flow_res_{val_id}.png')

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            if occlu_root:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        # px1 = np.mean(epe_all<1)
        # px3 = np.mean(epe_all<3)
        # px5 = np.mean(epe_all<5)

        # print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype+'_all'] = np.mean(epe_list)

        if occlu_root:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))
            results[dstype + '_matched'] = matched_epe
            results[dstype + '_unmatched'] = unmatched_epe
            print('===> Validatation Sintel (%s): all epe: %.3f matched epe: %.3f, unmatched epe: %.3f' % (
                dstype, epe, matched_epe, unmatched_epe))
        else:
            print('===> Validatation Sintel (%s): all epe: %.3f' % (dstype, epe))


    with open('logs/test/sintel_inr.txt', 'w') as f:
        f.write(str(results))

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--image_root', help="images in dataset")
    parser.add_argument('--flow_root', help="flows in dataset")
    parser.add_argument('--occlu_root', default=None, help="occlusion maps in dataset")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    cfg = get_cfg()
    # if args.small:
    #     cfg = get_small_things_cfg()
    # else:
    #     cfg = get_things_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(args)
    os.makedirs('logs/test/clean', exist_ok=True)
    os.makedirs('logs/test/final', exist_ok=True)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'sintel_inr':
            validate_sintel_inr(model.module, image_root=cfg.image_root, flow_root=cfg.flow_root, occlu_root=cfg.occlu_root)



