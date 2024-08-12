import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat
from mmcv.cnn import get_model_complexity_info
import logging
import warnings

from PIL import Image
from torchvision import transforms
import numpy as np

# Assuming these functions are defined elsewhere in your code
from dataset.utils import scale_aligned_short

from dataset.utils import get_img

warnings.filterwarnings('ignore')


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|') 
    print('-' * 90)
    num_para = 0
    type_size = 1 

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def prepare_test_data(img_path, short_size, read_type):
    filename = img_path.split('/')[-1][:-4]

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
    print(data)
    return data


def main(args):
    cfg = Config.fromfile(args.config)

    # Model
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
    model = fuse_module(model)

    if args.print_model:
        model_structure(model)

    model.eval()

    # Single image inference
    image_path = args.image_path

    # Use prepare_test_data for image preprocessing
    data = prepare_test_data(image_path, cfg.data.test.short_size, cfg.data.test.read_type)

    # Convert img_metas to the format expected by the model
    data['img_metas'] = {'filename': data['img_metas']['filename'], 
                         'org_img_size': data['img_metas']['org_img_size'],
                         'img_size': data['img_metas']['img_size']}
    
    data['imgs'] = data['imgs'][None].cuda() if not args.cpu else data['imgs'][None]
    data.update(dict(cfg=cfg))

    with torch.no_grad():
        outputs = model(**data)

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)
    print(outputs['results'][0])
    print(f"Inference results for {image_path} saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--image-path', type=str, help='Path to a single image for inference', required=True)

    args = parser.parse_args()
    mmcv.mkdir_or_exist("./speed_test") 
    config_name = os.path.basename(args.config)
    logging.basicConfig(filename=f'./speed_test/{config_name}.txt', level=logging.INFO)

    main(args)
