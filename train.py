import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
imdb_train_name = 'caltech_train_1x'
imdb_val_name = 'caltech_val_1x'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_models/VGG_imagenet.npy'
output_dir = 'models/saved_models'

lr_decay = 1. / 10

rand_seed = 42
_DEBUG = True
use_tensorboard = True
remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard

ADAM = False
QUICK = True

start_step = 0
if QUICK:
    end_step = 160000
    lr_decay_steps = {80000, 120000, 140000}
else:
    end_step = 600000
    lr_decay_steps = {100000, 200000, 400000}

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb_train = get_imdb(imdb_train_name)
imdb_val = get_imdb(imdb_val_name)
rdl_roidb.prepare_roidb(imdb_train)
roidb_train = imdb_train.roidb
rdl_roidb.prepare_roidb(imdb_val)
roidb_val = imdb_val.roidb
data_layer_train = RoIDataLayer(roidb_train, imdb_train.num_classes)
data_layer_val = RoIDataLayer(roidb_val, imdb_val.num_classes)

# load net
net = FasterRCNN(classes=imdb_train.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy(net, pretrained_model)

net = net.cuda()
# net = torch.nn.DataParallel(net)
net.train()

params = list(net.parameters())


def _make_optimiser():
    if ADAM:
        optimizer = torch.optim.Adam(params[8:], lr=lr)
    else:
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


optimizer = _make_optimiser()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        datetime = datetime.now().strftime(' %m/%d-%H:%M')
        net_name = pretrained_model.split('/')[-1].split('.')[0]
        pace = " quick" if QUICK else " slow"
        exp_name = "Faster RCNN " + net_name + pace + datetime
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tn, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer_train.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tn += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TN: %.2f%%, fg/bg=(%d/%d)' % (
                tp / fg * 100., tn / bg * 100., fg / step_cnt, bg / step_cnt),
                      color='cyan')
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0],
                net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0],
                net.loss_box.data.cpu().numpy()[0]),
                      color='white')
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        # Validation
        val_loss = 0
        val_size = 5
        for _ in range(val_size):
            # get one batch
            blobs = data_layer_val.forward()
            im_data = blobs['data']
            im_info = blobs['im_info']
            gt_boxes = blobs['gt_boxes']
            gt_ishard = blobs['gt_ishard']
            dontcare_areas = blobs['dontcare_areas']

            # forward
            net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
            loss = net.loss + net.rpn.loss
            val_loss += loss.data[0]
        val_loss = val_loss / val_size

        exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
        exp.add_scalar_value('val_loss', val_loss, step=step)
        exp.add_scalar_value('learning_rate', lr, step=step)
        if _DEBUG:
            exp.add_scalar_value('true_positive', tp / fg * 100., step=step)
            exp.add_scalar_value('true_negative', tn / bg * 100., step=step)
            losses = {
                'rpn_cls_cross_entropy':  float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                'rpn_box_loss':           float(net.rpn.loss_box.data.cpu().numpy()[0]),
                'rcnn_cls_cross_entropy': float(net.cross_entropy.data.cpu().numpy()[0]),
                'rcnn_box_loss':          float(net.loss_box.data.cpu().numpy()[0])}
            exp.add_scalar_dict(losses, step=step)

    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = _make_optimiser()

    if re_cnt:
        tp, tn, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
