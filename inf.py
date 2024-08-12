import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat
from mmcv.cnn import get_model_complexity_info
import logging
import warnings
warnings.filterwarnings('ignore')

def inference(image_path, model, cfg, out_dir='outputs/', save_pred=True, save_vis=True):
    # data loader for a single image
    data_loader = build_data_loader(cfg.data.test, image_path)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,  # Single image inference
        shuffle=False,
        num_workers=0, # No need for multiple workers for single image
        pin_memory=False
    )

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    for idx, data in enumerate(test_loader):
        if not args.cpu:
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        data.update(dict(cfg=cfg))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        # save result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            rf.write_result(image_name, outputs['results'][index])
            if save_vis:
                img = mmcv.imread(image_path)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                bboxes = outputs['results'][index]
                mmcv.imshow_det_bboxes(
                    img.copy(),
                    bboxes,
                    show=True,
                    score_thr=cfg.test_cfg.min_score,
                    bbox_color=cfg.test_cfg.bbox_color,
                    text_color=cfg.test_cfg.text_color,
                    thickness=cfg.test_cfg.thickness,
                    font_size=cfg.test_cfg.font_size,
                    win_name='',
                    out_file= os.path.join(out_dir, image_name)) 
                
def main(args):
    cfg = Config.fromfile(args.config)

    if args.min_score is not None:
        cfg.test_cfg.min_score = args.min_score
    if args.min_area is not None:
        cfg.test_cfg.min_area = args.min_area

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
    model = fuse_module(model)
    
    if args.print_model:
        model_structure(model)

    model.eval()
    
    # Inference on the specified image
    inference(args.image_path, model, cfg, args.out_dir, args.save_pred, args.save_vis) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--out_dir', type=str, default='outputs/', help='Output directory for saving results')
    parser.add_argument('--save_pred', action='store_true', help='Save predicted bounding boxes')
    parser.add_argument('--save_vis', action='store_true', help='Save visualized image with bounding boxes')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    mmcv.mkdir_or_exist(args.out_dir)
    config_name = os.path.basename(args.config)
    logging.basicConfig(filename=f'./{config_name}.txt', level=logging.INFO)

    main(args)
