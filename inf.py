import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from mmcv.cnn import get_model_complexity_info
import logging
import warnings
warnings.filterwarnings('ignore')
import json
from dataset.utils import get_img
from dataset.utils import scale_aligned_short
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


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

    return data


def inf(inf_data, model, cfg):

    rf = ResultFormat(cfg.data.test.type, "inference_output")

    results = dict()
    
    print('Testing the image...', flush=True, end='') 

    if not args.cpu:
        inf_data['imgs'] = inf_data['imgs'].unsqueeze(0).cuda(non_blocking=True)
        inf_data['img_metas'] = inf_data['img_metas'].unsqueeze(0)
    inf_data.update(dict(cfg=cfg))

    with torch.no_grad():
        outputs = model(**inf_data)

    print(outputs['results'][0])



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
    inf_data = prepare_inf_data(args.img_path)
    inf(inf_data, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', nargs='?', type=str, default=None)
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('img_path', nargs='?', type=str, default=None)
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    main(args)
