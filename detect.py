import argparse
import os
import cv2
import cPickle

import datetime
import time
import numpy as np
import sys

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

# hyper-parameters
# ------------
model_dir = 'models/saved_models/'
imdb_name = 'detection'
cfg_file = os.path.join(model_dir, 'caltech.yml')
trained_model = os.path.join(model_dir, 'faster_rcnn_140000.h5')

rand_seed = 42

save_name = 'faster_rcnn_100000'
max_per_image = 300
thresh = 5e-5
vis = False
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


def vis_detections(im, class_name, dets, thresh=0.6):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    im_info = np.array([[im_data.shape[1], im_data.shape[2], im_scales[0]]], dtype=np.float32)

    cls_prob, bbox_pred, rois = net(im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def detect(name, net, imdb, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, name)

    # timers
    det_file = os.path.join(output_dir, 'detections.pkl')

    avg_fps = 5
    last_time = time.time() - 1
    display_freq = 10
    if vis:
        display_freq = 1
    bar_length = 50
    for i in range(num_images):

        im = cv2.imread(imdb.image_path_at(i))
        scores, boxes = im_detect(net, im)

        if vis:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32,
                                                                                copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if i % display_freq == 0:
            percent = float(i + 1) / num_images
            arrow = '-' * int(round(percent * bar_length) - 1) + '>'
            spaces = ' ' * (bar_length - len(arrow))

            now = time.time()
            fps = float(display_freq) / (now - last_time)
            last_time = now
            avg_fps = avg_fps * (1 - 0.01 * display_freq) + fps * 0.01 * display_freq
            seconds_remaining = datetime.timedelta(seconds=int((num_images - i) / avg_fps))
            sys.stdout.write("\r" + " " * (bar_length + 100))
            sys.stdout.flush()
            sys.stdout.write("\rTesting: [{}] {:3d}%; {:3.2f} FPS; Time remaining: {}".
                             format(arrow + spaces, int(percent * 100), avg_fps, seconds_remaining))
            sys.stdout.flush()

            if vis:
                cv2.imshow('test', im2show)
                cv2.waitKey(1)
    print '\n'

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.write_results(all_boxes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect people in video')

    parser.add_argument('--update', action='store_true')
    parser.add_argument('--fps', default=25)
    args = parser.parse_args()

    if args.update:
        print "Converting video:",
        # Currently can convert video up to an hour long at 25FPS
        os.system("cd ~/pedestrian-detection/detection/ && rm frames/*; rm results/*;"
                  "ffmpeg -loglevel panic -i input.mov -r {} -q:v 2 frames/%05d.jpg".
                  format(args.fps))
        os.system("cd ~/pedestrian-detection/detection/ && python make_frame_list.py")
        print("Done")
    # load data
    imdb = get_imdb(imdb_name)

    # load net
    net = FasterRCNN(classes=imdb.classes, debug=False, cfg=cfg)
    network.load_net(trained_model, net)
    print('load model successfully!')

    net.cuda()
    net.eval()

    # detect people in each frame
    detect(save_name, net, imdb, max_per_image, thresh=thresh, vis=vis)
    # Convert frames to video
    os.system("cd ~/pedestrian-detection/detection/ &&"
              " ffmpeg -framerate {} -i results/%05d.jpg output-{}FPS.mov".format(args.fps,
                                                                                  args.fps))
