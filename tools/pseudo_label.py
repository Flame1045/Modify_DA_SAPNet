#!python3

import os
import sys
import time
import json
import gzip
import random
import tqdm
import argparse
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
# 100 IDs in Scenes100
type_list = ['cityscapes']

# 2 target classes
thing_classes = ['car']

# identification color
bbox_rgbs = ['#FF0000', '#0000FF']

# mapping from MSCOCO classes to 2 target classes
thing_classes_coco = [['person'], ['car', 'bus', 'truck']]

def get_cfg_base_model(m, ckpt=None):
    models = {
        'r50-fpn-3x': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'r101-fpn-3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    }
    assert m in models, 'model %s not recognized' % m

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(models[m]))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = ckpt
        cfg.MODEL.WEIGHTS = os.path.normpath(cfg.MODEL.WEIGHTS)

    print('detectron2 model:', m)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    return cfg

def detect():
    import detectron2
    from detectron2.engine import DefaultPredictor
    
    imagedir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages'
    with open('/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/ImageSets/Main/train.txt', 'r') as fp:
        ifilelist = fp.readlines() 
        # print(ifilelist)

    cfg = get_cfg_base_model(args.model, ckpt=args.ckpt)
    predictor = DefaultPredictor(cfg)
    print('detect objects with %s in video %s %s' % (args.model, args.id, imagedir), flush=True)
    iamges_list = []
    # images_dict = {}
    categories_list = []
    categories_dict = {}
    annotations = []
    image_id = 2000
    k_id = 5000
    for f in tqdm.tqdm(ifilelist, ascii=True):
        f = f[:-1] + '.jpg'
        print(f)
        iamges_list.append({
            'id': image_id + 1,
            'width': 2048,
            'height': 1024,
            'file_name': f
        })
        # images_dict["width"] = 2048
        # images_dict["height"] = 1024
        # images_dict["file_name"] = f
        # iamges_list.append(images_dict)
        im = detectron2.data.detection_utils.read_image(os.path.join(imagedir, f), format='BGR')
        instances = predictor(im)['instances'].to('cpu')
        bboxes = instances.pred_boxes.tensor.numpy().tolist()
        k_id_ = 0
        for k in tqdm.tqdm(bboxes, ascii=True):
            annotations.append({
                # bbox has format [x1, y1, x2, y2]
                'id': k_id + 1,
                'image_id': image_id + 1,
                'category_id': 1,
                'bbox': [int(x) for x in instances.pred_boxes.tensor.numpy().tolist()[k_id_]],
                'bbox_mode': BoxMode.XYXY_ABS,
                'score': instances.scores.numpy().tolist()[k_id_],
                'label': instances.pred_classes.numpy().tolist()[k_id_]
            })
            k_id = k_id + 1
            k_id_ = k_id_ + 1
        image_id = image_id + 1 
        if image_id == 2030:
            break
    # frame_objs.append({'category_id': 1})
    categories_list.append({
            'id': 1,
            'name': "car",
        })
    # categories_dict['id'] = 1
    # categories_dict['name'] = "car"
    # categories_list.append(categories_dict)
    result_json_zip = os.path.join(args.outputdir, '%s_detect_%s_test3.json.gz' % (args.id, args.model))
    with gzip.open(result_json_zip, 'wt') as fp:
        # fp.write(json.dumps({'model': args.model, 'classes': thing_classes, 'frames': ifilelist, 'dets': frame_objs, 'args': vars(args)}))
        fp.write(json.dumps({'images': iamges_list, 'categories': categories_list, 'annotations': annotations}))
    print('results saved to:', result_json_zip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pseudo Labeling Script')
    parser.add_argument('--opt', type=str, choices=['detect', 'track'], help='option')
    parser.add_argument('--id', type=str, choices=type_list, help='video ID')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x'], help='detection model')
    parser.add_argument('--ckpt', type=str, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.', help='save detection and tracking results to this directory')

    parser.add_argument('--pytracking_dir', type=str, help='root directory of PyTracking repository')
    parser.add_argument('--cuda_dir', type=str, help='root directory of CUDA toolkit')
    parser.add_argument('--detect_file', type=str, help='file that contains detected pseudo bounding boxes')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=50, help='minimum detection box size to start tracking')
    parser.add_argument('--sot_skip', type=float, default=-1, help='interval of video segments for tracking in seconds')
    parser.add_argument('--sot_max_length', type=float, default=2, help='maximum seconds of tracks')
    args = parser.parse_args()
    print(args)

    if args.opt == 'detect':
        detect()


'''
python pseudo_label.py --opt detect --id 001 --model r50-fpn-3x
python pseudo_label.py --opt track --id 001 --model r101-fpn-3x --sot_skip 5 --sot_max_length 2

python pseudo_label.py --opt detect --id cityscapes --model r101-fpn-3x 
--ckpt /media/ee4012/Disk3/Eric/scenes100/mscoco/models/mscoco2017_remap_r101-fpn-3x.pth --outputdir ./out22
'''
