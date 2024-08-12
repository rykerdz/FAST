import torch
from mmcv import Config
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from mmcv.parallel import collate
from mmdet.datasets.pipelines import Compose

class SingleImageDataset(Dataset):
    def __init__(self, image_path, pipeline):
        self.image_path = image_path
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return 1  # Single image

    def __getitem__(self, index):
        # Load the image and apply transformations
        data = dict(img_info=dict(filename=self.image_path), img_prefix=None)
        data = self.pipeline(data)
        return data

def infer(image_path, model, cfg):
    # Data loader for single image inference
    test_pipeline = cfg.data.test.pipeline
    dataset = SingleImageDataset(image_path, test_pipeline)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate  # Use the default collate function from mmcv
    )

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    # Inference
    for idx, data in enumerate(test_loader):
        if not args.cpu:
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        data.update(dict(cfg=cfg))
        with torch.no_grad():
            outputs = model(**data)

        # Save or display the result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            print(outputs['results'][index])
            print(f"Inference results for {image_name} saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('image_path', help='path to the image for inference')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.batch_size = 1  # Single image inference

    # Model
    model = build_model(cfg.model)
    if not args.cpu:
        model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            state_dict = checkpoint['state_dict']  # Assuming no EMA for inference

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
    model.eval()

    infer(args.image_path, model, cfg)
