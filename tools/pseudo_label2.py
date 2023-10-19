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
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import verify_results
from detectron2.utils.events import EventStorage, TensorboardXWriter
import os
import sys
from datetime import datetime
from pathlib import Path
import torch, gc
import random
import numpy as np
import cv2 
 
gc.collect()
torch.cuda.empty_cache()
sys.path.append(os.getcwd())

from detection.trainer import (
    DATrainer, 
    DefaultTrainer_, 
    SpatialAttentionVisualHelper, 
    GramCamForDomainClassfier,
    GramCamForObjectDetection,
)

from detection.hooks import EvalHook_

# register datasets
import detection.data.register

# register compoments
import detection.modeling
from detection.meta_arch.sap_rcnn import SAPRCNN
from detection.da_heads import build_DAHead
from detection.modeling.rpn import SAPRPN




def add_saprcnn_config(cfg):
    from detectron2.config import CfgNode as CN
    _C = cfg
    _C.MODEL.DOMAIN_ADAPTATION_ON = False
    _C.MODEL.DA_HEAD = CN()
    _C.MODEL.DA_HEAD.IN_FEATURE = "res4"
    _C.MODEL.DA_HEAD.IN_CHANNELS = 256
    _C.MODEL.DA_HEAD.NUM_ANCHOR_IN_IMG = 5
    _C.MODEL.DA_HEAD.EMBEDDING_KERNEL_SIZE = 3
    _C.MODEL.DA_HEAD.EMBEDDING_NORM = True
    _C.MODEL.DA_HEAD.EMBEDDING_DROPOUT = True
    _C.MODEL.DA_HEAD.FUNC_NAME = 'cross_entropy'
    _C.MODEL.DA_HEAD.POOL_TYPE = 'avg'
    _C.MODEL.DA_HEAD.LOSS_WEIGHT = 1.0
    _C.MODEL.DA_HEAD.WINDOW_STRIDES = [2, 2, 2]
    _C.MODEL.DA_HEAD.WINDOW_SIZES = [3, 6, 9]
    _C.MODEL.DA_HEAD.R = 1
    _C.MODEL.DA_HEAD.ALPHA = 'ones'
    _C.MODEL.PROPOSAL_GENERATOR.NAME = "SAPRPN"

    _C.MODEL.DA_HEAD.NAME = 'SAPNetMSCAM'
    _C.MODEL.DA_HEAD.RPN_MEDM_ON = False
    _C.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT = 0.
    _C.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT = 0.
    ##########MIC############
    _C.MODEL.DA_HEAD.MIC_ON = False   
    _C.MODEL.DA_HEAD.MIC_RPN_CLS_WEIGHT = 0.
    _C.MODEL.DA_HEAD.MIC_RPN_LOC_WEIGHT = 0.
    _C.MODEL.DA_HEAD.MIC_CLS_WEIGHT = 0.
    _C.MODEL.DA_HEAD.MIC_BOX_REG_WEIGHT = 0.
    _C.MODEL.DA_HEAD.MASKING_BLOCK_SIZE = 32
    _C.MODEL.DA_HEAD.MASKING_RATIO = 0.5
    _C.MODEL.DA_HEAD.MASK_COLOR_JITTER_S = 0.2
    _C.MODEL.DA_HEAD.MASK_COLOR_JITTER_P = 0.2
    _C.MODEL.DA_HEAD.MASK_BLUR = True
    _C.MODEL.DA_HEAD.PIXEL_MEAN = [0.485, 0.456, 0.406]
    _C.MODEL.DA_HEAD.PIXEL_STD = [0.229, 0.224, 0.225] 
    _C.MODEL.DA_HEAD.TEACHER_ALPHA = 0.9
    _C.MODEL.DA_HEAD.PSEUDO_LABEL_THRESHOLD =0.7
    ##########MIC############
    _C.DATASETS.SOURCE_DOMAIN = CN()
    _C.DATASETS.SOURCE_DOMAIN.TRAIN = ()
    _C.DATASETS.TARGET_DOMAIN = CN()
    _C.DATASETS.TARGET_DOMAIN.TRAIN = ()
    _C.SOLVER.NAME = "default"

def check_cfg(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        assert cfg.MODEL.DA_HEAD.LOSS_WEIGHT > 0,  'MODEL.DA_HEAD.LOSS_WEIGHT must be greater than 0'
    
    if cfg.MODEL.DA_HEAD.RPN_MEDM_ON:
        assert cfg.MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT > 0, 'MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT must be greater than 0'
        assert cfg.MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT < 0, 'MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT must be smaller than 0'

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    

def setup(args):
    cfg = get_cfg()
    add_saprcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    now = datetime.now()

    if args.eval_all:
        cfg.OUTPUT_DIR = str(Path(cfg.OUTPUT_DIR) / 'eval')
    # if not args.resume:
    #     cfg.OUTPUT_DIR = './outputs/output-{}'.format(now.strftime("%y-%m-%d_%H-%M"))
    #     if args.setting_token:
    #         cfg.OUTPUT_DIR = './outputs/output-{}-{}'.format(args.setting_token, now.strftime("%y-%m-%d_%H-%M"))
    
    # if not (args.test_images or args.visualize_attention_mask or args.gcs or args.gct or args.gco):
    #     default_setup(cfg, args)
    # elif args.visualize_attention_mask or args.gcs or args.gct or args.gco or args.test_images:
    #     if args.gcs or args.gct: 
    #         assert args.attention_mask or args.backbone_feature, 'please determine which feature to visualize'
    #         assert cfg.MODEL.DOMAIN_ADAPTATION_ON, 'domain classfier is used, cfg.MODEL.DOMAIN_ADAPTATION_ON should be True'
    #     # set random seed
    #     rank = comm.get_rank()
    #     seed = cfg.SEED
    #     seed_all_rng(None if seed < 0 else seed + rank)
    return cfg

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
        'r101-fpn-3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        'custom': 'custom'

    }
    assert m in models, 'model %s not recognized' % m

    cfg = setup(args)
    setup_seed(cfg.SEED)

    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file(models[m]))
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    if not ckpt is None:
        assert os.access(ckpt, os.R_OK), '%s not readable' % ckpt
        cfg.MODEL.WEIGHTS = ckpt
        cfg.MODEL.WEIGHTS = os.path.normpath(cfg.MODEL.WEIGHTS)
    cfg.freeze()
    check_cfg(cfg)
    print('detectron2 model:', m)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    print(cfg)
    return cfg

# def main(args):
    

    # print("train_net main seeding")

    # if args.eval_only:
    #     model = DATrainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = DATrainer.test(cfg, model)
    #     if comm.is_main_process():
    #         verify_results(cfg, res)
    #     return res

def detect():
    import detectron2
    from detectron2.engine import DefaultPredictor
    
    imagedir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages'
    with open('/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/ImageSets/Main/val.txt', 'r') as fp:
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
        # im = cv2.imread(os.path.join(imagedir, f), cv2.IMREAD_COLOR)
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
    parser = default_argument_parser()
    parser.add_argument('--opt', type=str, choices=['detect', 'track'], help='option')
    parser.add_argument('--id', type=str, choices=type_list, help='video ID')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x', 'custom'], help='detection model')
    parser.add_argument('--ckpt', type=str, help='weights checkpoint of model')
    parser.add_argument('--outputdir', type=str, default='.', help='save detection and tracking results to this directory')

    parser.add_argument('--pytracking_dir', type=str, help='root directory of PyTracking repository')
    parser.add_argument('--cuda_dir', type=str, help='root directory of CUDA toolkit')
    parser.add_argument('--detect_file', type=str, help='file that contains detected pseudo bounding boxes')
    parser.add_argument('--sot_score_thres', type=float, default=0.9, help='minimum detection score to start tracking')
    parser.add_argument('--sot_min_bbox', type=int, default=50, help='minimum detection box size to start tracking')
    parser.add_argument('--sot_skip', type=float, default=-1, help='interval of video segments for tracking in seconds')
    parser.add_argument('--sot_max_length', type=float, default=2, help='maximum seconds of tracks')
    parser.add_argument("--setting-token", help="add some simple profile about this experiment to output directory name")
    parser.add_argument("--test-images", action="store_true", help="output predicted bbox to test images")
    parser.add_argument("--eval-all", action="store_true", help="eval all checkpoint under the cfg.OUTPUT_DIR, and put result to its sub dir")

    # parser.add_argument('--config-file', type=str)
    args = parser.parse_args()
    print(args)

    if args.opt == 'detect':
        detect()


'''
python pseudo_label.py --opt detect --id 001 --model r50-fpn-3x
python pseudo_label.py --opt track --id 001 --model r101-fpn-3x --sot_skip 5 --sot_max_length 2

python pseudo_label.py --opt detect --id cityscapes --model r101-fpn-3x --ckpt /media/ee4012/Disk3/Eric/scenes100/mscoco/models/mscoco2017_remap_r101-fpn-3x.pth --outputdir ./out22
python tools/pseudo_label2.py --opt detect --id cityscapes --model custom --ckpt /media/ee4012/Disk3/Eric/Modify_DA_SAPNet/pretrained/sim2city-sapnetv1/model_0005199_MAP_50_64.pth --outputdir ./tools/out22 --config-file "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" --eval-all 
'''
