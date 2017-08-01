#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import json
from scipy.io import loadmat
from collections import defaultdict

import xml.etree.ElementTree as ET
import cPickle
import numpy as np


def parse_caltech_annotations(image_identifiers, ann_dir):
    # recs is a dictionary with keys as image_identifier.
    # value is a list of dictionaries where each dictionary belongs
    # to an object
    # Inside each dictionary the keys are 'name', 'bbox' etc
    recs = {}
    all_obj = 0
    data = defaultdict(dict)
    image_wd = 640
    image_ht = 480

    # Parse all the annotations and store
    for dname in sorted(glob.glob(ann_dir + '/set*')):
        set_name = os.path.basename(dname)
        data[set_name] = defaultdict(dict)
        for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
            vbb = loadmat(anno_fn)
            objLists = vbb['A'][0][0][1][0]
            objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]

            video_name = os.path.splitext(os.path.basename(anno_fn))[0]
            for frame_id, obj in enumerate(objLists):
                objs = []
                if len(obj) > 0:
                    for id, pos in zip(obj['id'][0], obj['pos'][0]):
                        id = int(id[0][0]) - 1  # MATLAB is 1-origin
                        keys = obj.dtype.names
                        pos = pos[0].tolist()
                        # convert x,y,w,h to xmin,ymin,xmax,ymax and clip to image frame
                        pos[0] = np.clip(pos[0], 0, image_wd)
                        pos[1] = np.clip(pos[1], 0, image_ht)
                        pos[2] = np.clip(pos[0] + pos[2], 0, image_wd)
                        pos[3] = np.clip(pos[1] + pos[3], 0, image_ht)
                        datum = dict(zip(keys, [id, pos]))
                        obj_datum = dict()
                        obj_datum['name'] = str(objLbl[datum['id']])
                        if obj_datum['name'] != 'person':
                            continue
                        obj_datum['pose'] = 'Unspecified'
                        obj_datum['truncated'] = 0
                        obj_datum['difficult'] = 0
                        obj_datum['bbox'] = pos
                        objs.append(obj_datum)
                data[set_name][video_name][frame_id] = objs

    # Out of all available annotations, just use those that are
    # required (as listed in image_identifiers)
    for image_identifier in image_identifiers:
        image_set_name = image_identifier[0:5]
        image_seq_name = image_identifier[6:10]
        image_id = int(image_identifier[11:])
        if image_id in data[image_set_name][image_seq_name]:
            recs[image_identifier] = data[image_set_name][image_seq_name][image_id]
        else:
            print
            "Warning: No %s.jpg found in annotations" % (image_identifier)

            # vis_annotations(image_identifier, recs[image_identifier])
    return recs


def parse_new_annotations(image_identifiers, ann_dir):
    recs = {}
    image_wd = 640
    image_ht = 480

    ann_dir = os.path.join(ann_dir, 'new_train_10x')

    for img in image_identifiers:
        path_split = img.split('/')
        name = path_split[0] + '_' + path_split[1] + '_' + 'I' + path_split[2].zfill(5) + '.txt'
        with open(os.path.join(ann_dir, name), 'r') as file:
            detections = file.readlines()[1:]
            objs = []
            for detection in detections:
                det = detection.split(' ')
                type = det[0]
                pos = np.array(det[1:5]).astype(np.float)
                bbox = np.zeros(4)
                # convert x,y,w,h to xmin,ymin,xmax,ymax
                bbox[0] = np.clip(pos[0], 0, image_wd)
                bbox[1] = np.clip(pos[1], 0, image_ht)
                bbox[2] = np.clip(pos[0] + pos[2], 0, image_wd)
                bbox[3] = np.clip(pos[1] + pos[3], 0, image_ht)
                obj_datum = dict()
                obj_datum['name'] = type
                obj_datum['bbox'] = bbox
                obj_datum['difficult'] = 0
                objs.append(obj_datum)
            recs[img] = objs
    return recs


def vis_annotations(image_identifier, dets):
    """Draw detected bounding boxes."""
    import cv2
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    im = cv2.imread(os.path.join('/media/disk2/govind/work/dataset/caltech/data/JPEGImages',
                                 image_identifier + '.jpg'))
    inds = dets
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for obj in inds:
        bbox = obj['bbox']
        # print bbox
        class_name = obj['name']
        ax.add_patch(
            plt.Rectangle(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                          bbox[2] - bbox[0], bbox[3] - bbox[1],
                          fill=False, edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_identifier + '_ann.jpg')


def caltech_ap(rec, prec, use_07_metric=False):
    """ ap = caltech_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def caltech_eval(detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(image_identifier) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(image_identifier)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    image_identifiers = [x.strip() for x in lines]

    if 1:  # load new anotations unconditionally in case test data have changed
        # load annots
        # govind: recs is a dictionary with <image_identifier> as keys
        recs = parse_caltech_annotations(image_identifiers, annopath)
        # save
        print
        'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    # govind: recs is not class specific. Hence create another
    # dictionary class_recs which is specific to this class
    class_recs = {}
    npos = 0
    nimg = len(image_identifiers)
    for image_identifier in image_identifiers:
        R = [obj for obj in recs[image_identifier] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[image_identifier] = {'bbox':      bbox,
                                        'difficult': difficult,
                                        'det':       det}

    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]  # Image name
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(confidence)

    if len(BB) != 0:  # check if array is empty
        BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    tot_tp = 0.
    tot_fp = 0.
    MR = np.ones(nd)
    FPPI = np.zeros(nd)
    for d in range(nd):
        # load all ground truths for this image
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    tot_tp += 1.
                else:
                    fp[d] = 1.
                    tot_fp += 1.
        else:
            fp[d] = 1.
            tot_fp += 1.

        MR[d] = 1. - tot_tp / npos
        FPPI[d] = tot_fp / nimg

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = caltech_ap(rec, prec, use_07_metric)

    return rec, prec, ap, MR, FPPI
