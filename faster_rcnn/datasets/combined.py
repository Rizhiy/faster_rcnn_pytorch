# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import datetime
import json

import matplotlib.pyplot as plt
import os

from faster_rcnn.datasets.imdb import imdb
import numpy as np
import scipy.sparse
import cPickle
import subprocess
import uuid
from caltech_utils import caltech_eval, parse_caltech_annotations, parse_new_annotations
from faster_rcnn.fast_rcnn.config import cfg


class combined(imdb):
    def __init__(self, devkit_path=None):
        # This is a combined dataset of ETH, TUD-Brussels and caltech_new_train_1x
        imdb.__init__(self, 'combined')
        self._image_set = 'combined'
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'person', 'ignore')

        # govind: num_classes is set based on the number of classes in _classes tuple
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):  # i is a number
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._data_path, 'images',
                                  self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        # govind: Returns a list of all image names
        return image_index


    def _get_default_path(self):
        """
        Return the default path where Caltech is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'caltech')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        else:
            imagesetfile = os.path.join(self._data_path, 'ImageSets',
                                        self._image_set + '.txt')

        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        caltech_parsed_data = parse_new_annotations(imagenames,
                                                    os.path.join(self._data_path, 'annotations'),
                                                    'combined')

        gt_roidb = [self._load_caltech_annotation(caltech_parsed_data, i)
                    for i in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_caltech_annotation(self, caltech_parsed_data, idx_image):
        """
        Load image and bounding boxes info from .vbb file in Caltech format.
        """
        # objs is a list of dictionaries. Each dictionary represents an object
        objs = caltech_parsed_data[idx_image]
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self._class_to_ind[obj['name']]
            boxes[ix, :] = obj['bbox']
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes':       boxes,
                'gt_classes':  gt_classes,
                'gt_overlaps': overlaps,
                'flipped':     False}
