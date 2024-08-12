
import cv2
import mmcv
import random
import argparse
import numpy as np
from PIL import Image


def get_pred(pred_path):
    lines = mmcv.list_from_file(pred_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        bbox = [int(gt[i]) for i in range(len(gt))]
        bboxes.append(bbox)
        words.append('???')
    return np.array(bboxes), words


def draw(img, boxes, words):
    
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    for box in boxes:
        rand_r = random.randint(100, 255)
        rand_g = random.randint(100, 255)
        rand_b = random.randint(100, 255)
        mask = cv2.fillPoly(mask, [box], color=(rand_r, rand_g, rand_b))
    
    img[mask!=0] = (0.6 * mask + 0.4 * img).astype(np.uint8)[mask!=0]
    
    for box, word in zip(boxes, words):
        if word == '###':
            cv2.drawContours(img, [box], -1, (255, 0, 0), thickness[args.dataset])
        else:
            cv2.drawContours(img, [box], -1, (0, 255, 0), thickness[args.dataset])

    return img
    
    
def visual(get_ann, data_dir, pred_dir, dataset):
    
    img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.JPG')])
    
    img_paths, pred_paths = [], []
    
    for idx, img_name in enumerate(img_names):
        img_path = data_dir + img_name
        img_paths.append(img_path)
        
            
    for index, (img_path, pred_path) in tqdm(enumerate(zip(img_paths, pred_paths)), total=len(img_paths)):
        img = get_img(img_path) # load image
        
        # load predictions
        pred, _ = get_pred(pred_path)
      
        pred = pred.tolist()
        for i in range(len(pred)):
            pred[i] = np.reshape(pred[i], (-1, 2)).astype('int32')
                
        img_pred = draw(img, pred, _) # draw predictions on images
        img = Image.fromarray(img)
        mmcv.mkdir_or_exist("inference_output")
        img.save(f"inference_output/{index}.png") # save images into inference output/
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('image_path', nargs='?', type=str, required=True)
    # show the ground truths with predictions
    args = parser.parse_args()
    
    # thickness for different datasets
    thickness = {'msra': 12, 'ctw':4, 'tt':4, 'ic15': 4}
    
    get_ann = get_ic15_ann
    test_data_dir = ic15_test_data_dir
    test_gt_dir = ic15_test_gt_dir
    pred_dir = ic15_pred_dir
        
    print(test_data_dir)
    visual(get_ann, test_data_dir, pred_dir, args.dataset)
