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
import torch

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

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


def refine_pseudo_labels(args):
    # imagedir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/JPEGImages'
    # labeldir = '/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/tools/'
    # det_filelist, sot_filelist = [], []
    # for m in ['r101-fpn-3x', 'r50-fpn-3x']:
    #     det_filelist.append(os.path.normpath(os.path.join(labeldir, 'cityscapes_detect_%s.json.gz' % (m))))
    #     # if not args.refine_no_sot:
    #     #     sot_filelist.append(os.path.normpath(os.path.join(labeldir, '%s_detect_%s_DiMP.json.gz' % (args.id, m))))
    # for f in det_filelist + sot_filelist:
    #     assert os.access(f, os.R_OK), '%s not readable' % f

    # collate bboxes from tracking & detection
    dicts_json = []
    annot_temp = []
    scores = []
    boxes = []
    # with open('/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/datasets/Cityscapes-coco/VOC2007-car-train/ImageSets/Main/train.txt', 'r') as fp:
    #     ifilelist = fp.readlines() 
    # for i in range(0, len(ifilelist)):
    #     dicts_json.append({'file_name': os.path.normpath(os.path.join(imagedir, ifilelist[i][:-1] + '.jpg')), 'image_id': i, 'height': 2048, 'width': 1024, 'annotations': [], 'det_count': 0, 'sot_count': 0, 'fn_count': 0})

    det_filelist = ['/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/tools/out22/cityscapes_detect_custom_test3.json']

    result_json_zip = os.path.join(args.outputdir, 'test.json.gz') 

    for jsons in det_filelist:
        with open(jsons, 'r') as f:
            # print('%s [%.2fMB]' % (f, os.path.getsize(f) / (1024 ** 2)))
            file = json.loads(f.read())
            annot = file['annotations']
            images = file['images']
            cat = file['categories']
            j = 0
            for i in range(0,len(annot)):
                if annot[i]['score'] < args.refine_det_score_thres:
                    continue
                annot_temp.append(annot[i])
                scores.append(annot_temp[j]['score'])
                boxes.append(annot_temp[j]['bbox'])
                j = j + 1
            ret = nms(boxes, scores)
            
            


            with gzip.open(result_json_zip, 'wt') as fp:
                fp.write(json.dumps({'images': images, 'categories': cat, 'annotations': annot_temp}))
   
    print('finish reading from detection results')
    # assert False

    # count_all = {'all': 0, 'det': 0, 'all_refined': 0}
    # for annotations in dicts_json:
    #     count_all['det'] += annotations['det_count']
    #     count_all['sot'] += annotations['sot_count']
    #     count_all['all'] += len(annotations['annotations'])
    # print('pseudo annotations: detection %d, tracking %d, total %d' % (count_all['det'], count_all['sot'], count_all['all']))



    ##############################
    # pool = ProcessPool(processes=os.cpu_count() // 3)
    # params_list, chunksize, i = [], len(dicts_json) // 20, 0
    # while True:
    #     dict_json_chunk = dicts_json[i * chunksize : (i + 1) * chunksize]
    #     if len(dict_json_chunk) < 1:
    #         break
    #     params_list.append({'dict': dict_json_chunk, 'args': args})
    #     i += 1
    # for i in range(0, len(params_list)):
    #     params_list[i]['desc'] = '%02d/%02d' % (i + 1, len(params_list))
    # refine_results = pool.map_async(_graph_refine, params_list).get()
    # pool.close()
    # pool.join()
    # dicts_json = []
    # for r in refine_results:
    #     dicts_json = dicts_json + r[0]
    #     count_all['all_refined'] += r[1]
    # print('%d images: refine pseudo bounding bboxes %d => %d (%.2f%%)' % (len(dicts_json), count_all['all'], count_all['all_refined'], count_all['all_refined'] / count_all['all'] * 100))
    # return dicts_json
    ##############################


if __name__ == '__main__':
    
    class DummyArgs(object):
        def __init__(self, args_dict):
            for k in args_dict:
                setattr(self, k, args_dict[k])

    results = refine_pseudo_labels(DummyArgs({
        'refine_det_score_thres': 0.98,
        'refine_iou_thres': 0.85,
        'outputdir': "./tools/out22"
    }))
