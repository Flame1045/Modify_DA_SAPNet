#!python3

import os
import sys
import glob
import json
import gzip
import random
import tqdm
from multiprocessing import Pool as ProcessPool
import numpy as np
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import networkx

sys.path.append(os.path.join(os.path.dirname(__file__)))
# from constants import thing_classes
def IoU(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    xA, yA = max(x11,x21), max(y11,y21)
    xB, yB = min(x12,x22), min(y12,y22)

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    overlap = max(xB - xA, 0) * max(yB - yA, 0)
    return overlap / (area1 + area2 - overlap)


def _graph_refine(params):
    _dicts_json, _args, _desc = params['dict'], params['args'], params['desc']
    count_bboxes = 0
    for annotations in tqdm.tqdm(_dicts_json, ascii=True, desc='refining chunk ' + _desc):
        G = networkx.Graph()
        [G.add_node(i) for i in range(0, len(annotations['annotations']))]
        for i in range(0, len(annotations['annotations'])):
            for j in range(i, len(annotations['annotations'])):
                iou_value = IoU(annotations['annotations'][i]['bbox'], annotations['annotations'][j]['bbox'])
                if annotations['annotations'][i]['category_id'] == annotations['annotations'][j]['category_id'] and iou_value > _args.refine_iou_thres:
                    G.add_edge(i, j, iou=iou_value)
        subs = list(networkx.algorithms.components.connected_components(G))

        anns_refine = []
        for sub_nodes in subs:
            max_degree, max_degree_n = -1, -1
            for n in sub_nodes:
                D = sum(map(lambda t: t[2], list(G.edges(n, data='iou'))))
                if D > max_degree:
                    max_degree, max_degree_n = D, n
            anns_refine.append(annotations['annotations'][max_degree_n])
        annotations['annotations'] = anns_refine
        if 'det_count' in annotations: del annotations['det_count']
        if 'sot_count' in annotations: del annotations['sot_count']
        annotations['bbox_count'] = len(annotations['annotations'])
        count_bboxes += annotations['bbox_count']
    return _dicts_json, count_bboxes


def refine_pseudo_labels(args):
    imagedir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages'
    labeldir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/tools/'
    det_filelist, sot_filelist = [], []
    for m in ['r101-fpn-3x', 'r50-fpn-3x']:
        det_filelist.append(os.path.normpath(os.path.join(labeldir, 'cityscapes_detect_%s.json.gz' % (m))))
        # if not args.refine_no_sot:
        #     sot_filelist.append(os.path.normpath(os.path.join(labeldir, '%s_detect_%s_DiMP.json.gz' % (args.id, m))))
    for f in det_filelist + sot_filelist:
        assert os.access(f, os.R_OK), '%s not readable' % f

    # collate bboxes from tracking & detection
    dicts_json = []
    # with open(os.path.join(imagedir, 'frames.json'), 'r') as fp:
    #     frames = json.load(fp)
    with open('/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/ImageSets/Main/train.txt', 'r') as fp:
        ifilelist = fp.readlines() 
    for i in range(0, len(ifilelist)):
        dicts_json.append({'file_name': os.path.normpath(os.path.join(imagedir, ifilelist[i][:-1] + '.jpg')), 'image_id': i, 'height': 2048, 'width': 1024, 'annotations': [], 'det_count': 0, 'sot_count': 0, 'fn_count': 0})
    # if 'fusion' in args.fusion:
    #     background_files = sorted(glob.glob(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scenes100', 'train_background', args.id, 'inpaint', '*inpaint.jpg'))))
    #     background_frame_idx = list(map(lambda x: os.path.basename(x), background_files))
    #     background_frame_idx = np.array(list(map(lambda x: int(x[:x.find('.')]), background_frame_idx)))
    #     for i in tqdm.tqdm(range(0, len(frames['ifilelist'])), ascii=True, desc='adding background images'):
    #         idx = os.path.basename(dicts_json[i]['file_name'])
    #         idx = int(idx[: idx.find('.')])
    #         dicts_json[i]['file_name_background'] = background_files[np.absolute(background_frame_idx - i).argmin()]

    for f in det_filelist:
        print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
        with gzip.open(f, 'rt') as fp:
            dets = json.loads(fp.read())['dets']
        assert len(dets) == len(dicts_json), 'detection & dataset mismatch'
        for i in range(0, len(dets)):
            for j in range(0, len(dets[i]['score'])):
                if dets[i]['score'][j] < args.refine_det_score_thres:
                    continue
                dicts_json[i]['annotations'].append({'bbox': dets[i]['bbox'][j], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': dets[i]['label'][j], 'src': 'det', 'score': dets[i]['score'][j]})
                dicts_json[i]['det_count'] += 1

    # for f in sot_filelist:
    #     print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
    #     with gzip.open(f, 'rt') as fp:
    #         _t = json.loads(fp.read())
    #         _forward, _backward = _t['forward'], _t['backward']
    #     assert len(_forward) == len(dicts_json) and len(_backward) == len(dicts_json), 'tracking & dataset mismatch'
    #     for i in range(0, len(_forward)):
    #         for tr in _forward[i]:
    #             dicts_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
    #             dicts_json[i]['sot_count'] += 1
    #     for i in range(0, len(_backward)):
    #         for tr in _backward[i]:
    #             dicts_json[i]['annotations'].append({'bbox': tr['bbox'], 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': tr['class'], 'src': 'sot', 'init_score': tr['init_score'], 'track_length': tr['track_length']})
    #             dicts_json[i]['sot_count'] += 1
    print('finish reading from detection & tracking results')

    # dicts_json = dicts_json[:len(dicts_json) // 20]
    # min_bbox_width, max_bbox_width = 5, 1000
    count_all = {'all': 0, 'det': 0, 'sot': 0, 'all_refined': 0}
    for annotations in dicts_json:
        count_all['det'] += annotations['det_count']
        count_all['sot'] += annotations['sot_count']
        count_all['all'] += len(annotations['annotations'])
        # annotations['annotations'] = list(filter(lambda ann: min(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) >= min_bbox_width, annotations['annotations']))
        # annotations['annotations'] = list(filter(lambda ann: max(ann['bbox'][2] - ann['bbox'][0], ann['bbox'][3] - ann['bbox'][1]) <= max_bbox_width, annotations['annotations']))
    print('pseudo annotations: detection %d, tracking %d, total %d' % (count_all['det'], count_all['sot'], count_all['all']))

    pool = ProcessPool(processes=os.cpu_count() // 3)
    params_list, chunksize, i = [], len(dicts_json) // 20, 0
    while True:
        dict_json_chunk = dicts_json[i * chunksize : (i + 1) * chunksize]
        if len(dict_json_chunk) < 1:
            break
        params_list.append({'dict': dict_json_chunk, 'args': args})
        i += 1
    for i in range(0, len(params_list)):
        params_list[i]['desc'] = '%02d/%02d' % (i + 1, len(params_list))
    refine_results = pool.map_async(_graph_refine, params_list).get()
    pool.close()
    pool.join()
    dicts_json = []
    for r in refine_results:
        dicts_json = dicts_json + r[0]
        count_all['all_refined'] += r[1]
    print('%d images: refine pseudo bounding bboxes %d => %d (%.2f%%)' % (len(dicts_json), count_all['all'], count_all['all_refined'], count_all['all_refined'] / count_all['all'] * 100))
    return dicts_json


def get_manual_dicts(video_id):
    inputdir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'scenes100', 'annotation', video_id))
    with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
        annotations = json.load(fp)
    for i in range(0, len(annotations)):
        annotations[i]['file_name'] = os.path.join(inputdir, 'unmasked', annotations[i]['file_name'])
        annotations[i]['image_id'] = i + 1
    print('manual annotation for %s: %d images, %d bboxes' % (video_id, len(annotations), sum(list(map(lambda x: len(x['annotations']), annotations)))))
    return annotations


if __name__ == '__main__':
    
    class DummyArgs(object):
        def __init__(self, args_dict):
            for k in args_dict:
                setattr(self, k, args_dict[k])

    # refine_pseudo_labels(DummyArgs({
    #     'id': '003',
    #     'anno_models': ['r50-fpn-3x'],
    #     'refine_det_score_thres': 0.5,
    #     'refine_iou_thres': 0.85,
    #     'refine_no_sot': False,
    #     'fusion': 'midfusion'
    # }))
    results = refine_pseudo_labels(DummyArgs({
        'id': '003',
        'anno_models': ['r50-fpn-3x', 'r101-fpn-3x'],
        'refine_det_score_thres': 0.90,
        'refine_iou_thres': 0.85,
        'refine_no_sot': True,
        'fusion': 'vanilla'
    }))

    import cv2

    for image in range(0,len(results)):
        print('image:', image)
        img = results[image]['file_name']
        img = cv2.imread(img)
        for bbox in range(0,len(results[image]['annotations'])):
            print('bbox:', bbox)
            x1, y1, x2, y2 = results[image]['annotations'][bbox]['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.imwrite('/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/test2/'+str(image)+'.jpg', img)
