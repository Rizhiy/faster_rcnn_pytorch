# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import os

from faster_rcnn.datasets.imdb import imdb
from PIL import Image, ImageDraw
from faster_rcnn.fast_rcnn.config import cfg


class detection(imdb):
    def __init__(self, data_path=None, draw_conf=True, conf=0.5):
        imdb.__init__(self, 'detect')
        self._image_set = 'detect'
        self._data_path = self._get_default_path() if data_path is None else data_path
        self._classes = ('__background__',  # always index 0
                         'person', 'ignore')
        # govind: num_classes is set based on the number of classes in _classes tuple
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self.conf = conf
        self.draw_conf = draw_conf

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):  # i is a number
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = os.path.join(self._data_path, 'frames',
                                  self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def results_path_at(self, i):
        image_path = os.path.join(self._data_path, 'results',
                                  self._image_index[i] + self._image_ext)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'image_set.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        # govind: Returns a list of all image names
        return image_index

    def write_results(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls != 'person':
                continue
            for im_ind, img_name in enumerate(self.image_index):
                dets = all_boxes[cls_ind][im_ind]
                img = Image.open(self.image_path_at(im_ind))
                draw = ImageDraw.Draw(img)
                if dets == []:
                    continue
                for i in range(dets.shape[0]):
                    conf = dets[i, -1]
                    if conf > self.conf:
                        if self.draw_conf:
                            draw.text(dets[i, 0:2], str(conf),fill='green')
                        # Right and bottom border might be off by one pixel
                        draw.rectangle(dets[i, 0:4],outline='greenyellow')
                del draw
                img.save(self.results_path_at(im_ind))

    def _get_default_path(self):
        """
        Return the default path where frames are expected to be found.
        """
        return os.path.join(cfg.DATA_DIR, 'detection')
