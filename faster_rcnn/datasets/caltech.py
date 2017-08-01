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


class caltech(imdb):
    def __init__(self, image_set, devkit_path=None):
        # govind: values of image_set is "train", "test" or "val"
        # govind: devkit_path is path to dataset directory

        imdb.__init__(self, 'caltech_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'person')

        # govind: num_classes is set based on the number of classes in _classes tuple
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._annotations = self._load_caltech_annotations()
        self._image_index = self._clean_image_index()

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'
        self.config['matlab_eval'] = False

        assert os.path.exists(self._devkit_path), \
            'Caltech devkit path does not exist: {}'.format(self._devkit_path)
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

    # govind: This function create an image_index object which has list of
    # all images for that particular _image_set
    # govind: This returns a a list of all image identifiers.
    # e.g. for image 1022.jpg in set01, V003 it stores "set01/V003/1022"
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

    def _load_caltech_annotations(self):
        annotations_file = os.path.join(self._data_path, 'annotations.json')
        assert os.path.exists(annotations_file), \
            'Path does not exist: {}'.format(annotations_file)
        annotations = json.load(open(annotations_file, 'r'))
        return annotations

    def _person_present(self, img):
        path = img.split('/')
        annotation = self._annotations[path[0]][path[1]]['frames']
        if path[2] not in annotation:
            return False
        for obj in annotation[path[2]]:
            if obj['lbl'] == 'person':
                return True
        return False

    def _clean_image_index(self):
        if 'train' in self._image_set or 'val' in self._image_set:
            image_index = [x for x in self._image_index if self._person_present(x)]
        else:
            # image_index = [x for idx, x in enumerate(self._image_index) if idx % 10 == 0]
            image_index = self._image_index
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
            print
            '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        else:
            imagesetfile = os.path.join(self._data_path, 'ImageSets',
                                        self._image_set + '.txt')

        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        if 'train_10x' in self._image_set:
            caltech_parsed_data = parse_new_annotations(imagenames,
                                                        os.path.join(self._data_path,
                                                                     'annotations'))
        else:
            caltech_parsed_data = parse_caltech_annotations(imagenames,
                                                            os.path.join(self._data_path,
                                                                         'annotations'))

        gt_roidb = [self._load_caltech_annotation(caltech_parsed_data, i)
                    for i in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print
        'wrote gt roidb to {}'.format(cache_file)
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

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_caltech_results_file_template(self):
        results_dir = path = os.path.join(
            self._devkit_path, 'results')
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        # caltech/results/<_get_comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        return os.path.join(results_dir, filename)

    def _write_caltech_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} caltech results file'.format(cls)
            filename = self._get_caltech_results_file_template().format(cls)

            lastset = None
            lastseq = None
            f = None
            # Caltech expects results for every sequence in a separate file with each set represented by a folder
            for im_ind, index in enumerate(self.image_index):
                idx_split = index.split('/')
                set = idx_split[0]
                if set != lastset:
                    dir = os.path.join(self._devkit_path, 'results', 'matlab_results', set)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    lastset = set
                seq = idx_split[1]
                if seq != lastseq:
                    if f:
                        f.close()
                    f = open(os.path.join(self._devkit_path, 'results', 'matlab_results', set,
                                          seq + '.txt'), 'w')
                    lastseq = seq
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for i in range(dets.shape[0]):
                    # Caltech expects data in form img,x,y,w,h,conf
                    x = (dets[i, 0] + dets[i, 2]) / 2
                    y = (dets[i, 1] + dets[i, 3]) / 2
                    w = dets[i, 2] - dets[i, 0]
                    h = dets[i, 3] - dets[i, 1]
                    x -= w / 2
                    y -= h / 2
                    f.write('{},{},{},{},{},{}\n'.format(idx_split[2], x, y, w, h, dets[i, -1]))

            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    # govind: This function is responsible for evaluating the
    # performance of network by comparing the
    # It is executed at the end of testing and is called by
    # lib/fast_rcnn/test_net()
    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._data_path, 'annotations')
        imagesetfile = os.path.join(self._data_path, 'ImageSets', self._image_set + '.txt')

        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        resultsdir = os.path.join(self._devkit_path, 'results')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # govind: Write Precision, Recall results of each class
        # into a separate .pkl file
        for i, cls in enumerate(self._classes):
            # govind: Ignore all other classes, including '__background__'
            if cls != 'person':
                continue
            filename = self._get_caltech_results_file_template().format(cls)
            # govind: It's calling a function which will give Precision, Recall and
            # average precision if we pass it the _results_file
            rec, prec, ap, mr, fppi = caltech_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=True)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # Caltech dataset reports performance in log-average miss rate on results with mr between 1e-2 and 0.
        mr4i = next(idx for idx, element in enumerate(fppi) if element > 1e-4)
        mr2i = next(idx for idx, element in enumerate(fppi) if element > 1e-2)
        try:
            mr0i = next(idx for idx, element in enumerate(fppi) if element > 1.)
        except StopIteration:
            mr0i = -1
        mr4 = mr[mr4i:mr0i]
        mr2 = mr[mr2i:mr0i]
        log_average_mr2 = np.exp(np.mean(np.log(mr2)))
        log_average_mr4 = np.exp(np.mean(np.log(mr4)))
        # log_average_mr2 = np.mean(mr2)
        # log_average_mr4 = np.mean(mr4)
        np.set_printoptions(precision=3)
        # govind: Computing Mean average precision
        # print("Miss rate: {}".format(mr))
        # print("FPPI:      {}".format(fppi))
        # print('~~~~~~~~')
        # print("LAMR: {} ({})".format(log_average_mr2, log_average_mr4))
        # print('Mean AP = {:.4f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('Results:')
        # for ap in aps:
        #     print('{:.3f}'.format(ap))
        # print('{:.3f}'.format(np.mean(aps)))
        # print('~~~~~~~~')
        # print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')

        print("CURRENTLY PYTHON TESTING DOESN'T WORK, USE MATLAB")

        now = str(datetime.datetime.now())
        plt.scatter(fppi, mr, label='{:3.3f} ({:3.3f})'.format(log_average_mr2, log_average_mr4),
                    s=2)
        plt.ylabel("Miss rate")
        plt.xlabel("FPPI")
        plt.yscale('log')
        plt.xlim([1e-4, 1e1])
        plt.ylim([0.04, 1.])
        plt.xscale('log')
        plt.title("Caltech results")
        plt.legend()
        plt.savefig(os.path.join(resultsdir, now + "-graph.png"),
                    orientation="landscape", bbox_inches='tight')

    def _do_matlab_eval(self):
        print
        '-----------------------------------------------------'
        print
        'Computing results with the official MATLAB eval code.'
        print
        '-----------------------------------------------------'
        path = os.path.join(self._devkit_path, 'evaluation_code')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop -r dbEval'.format(cfg.MATLAB)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
        print(status)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_caltech_results_file(all_boxes)
        # self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval()
        if self.config['cleanup']:
            # govind: Remove the temp result files
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_caltech_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
