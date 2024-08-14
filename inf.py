import torch
import argparse
import os
import sys
from mmcv import Config
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat
import logging
import warnings
warnings.filterwarnings('ignore')
from dataset.utils import get_img
from dataset.utils import scale_aligned_short
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
import cv2


# Draw function borrowed from visualize.py
def draw(img, boxes):

    for i in range(len(boxes)):
        boxes[i] = np.reshape(boxes[i], (-1, 2)).astype('int32')
    
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    for box in boxes:
        rand_r = random.randint(100, 255)
        rand_g = random.randint(100, 255)
        rand_b = random.randint(100, 255)
        mask = cv2.fillPoly(mask, [box], color=(rand_r, rand_g, rand_b))
    
    
    img[mask!=0] = (0.6 * mask + 0.4 * img).astype(np.uint8)[mask!=0]

    
    for box in boxes:
        cv2.drawContours(img, [box], -1, (0, 255, 0), 4)
    return img
    

# prepare_inf_data from FASTIC15
def prepare_inf_data(img_path, read_type="cv2", short_size=736):
    filename = img_path.split('/')[-1]

    img = get_img(img_path, read_type)
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    img = scale_aligned_short(img, short_size)
    img_meta.update(dict(
        img_size=np.array(img.shape[:2]),
        filename=filename
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    data = dict(
        imgs=img,
        img_metas=img_meta
    )
    img_to_draw = get_img(img_path)
    return data, img_to_draw


def inf(inf_data, img, model, cfg, save_to):
    
    print('Testing the image...', flush=True, end='') 

    # add batch dim
    inf_data['imgs'] = inf_data['imgs'].unsqueeze(0)
    if not args.cpu:
        inf_data['imgs'] = inf_data['imgs'].cuda(non_blocking=True)
    inf_data.update(dict(cfg=cfg))

    
    inf_data['img_metas'] = {
        'org_img_size': [inf_data['img_metas']['org_img_size'].tolist()],  # Convert to list of ints
        'img_size': [inf_data['img_metas']['img_size'].tolist()],        # Convert to list of ints
        'filename': [inf_data['img_metas']['filename']]                 # Wrap in a list (if it's not already)
    }

    
    with torch.no_grad():
        outputs = model(**inf_data)

    img_with_bboxes = draw(img, outputs['results'][0]['bboxes'])
    
    # save the image
    cv2.imwrite(save_to, img_with_bboxes)

    print(f"\n image saved to {save_to}")
    



def main(args):
    
    cfg = Config.fromfile(args.config)

    if args.min_score is not None:
        cfg.test_cfg.min_score = args.min_score
    if args.min_area is not None:
        cfg.test_cfg.min_area = args.min_area

    cfg.batch_size = 1

    # model
    model = build_model(cfg.model)
    
    if not args.cpu:
        model = model.cuda()
    
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            logging.info("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()
            checkpoint = torch.load(args.checkpoint)
            
            if not args.ema:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint['ema']

            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))
            raise
    
    model = rep_model_convert(model)

    # fuse conv and bn
    model = fuse_module(model)
    
    
    model.eval()
    inf_data, img = prepare_inf_data(args.img_path, cfg.data.test.read_type, cfg.data.test.short_size)
    
    inf(inf_data, img, model, cfg, args.save_to)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', nargs='?', type=str, default=None)
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('img_path', nargs='?', type=str, default=None)
    parser.add_argument('--save-to', nargs='?', type=str, default='output.jpg')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    main(args)
