
# coding: utf-8

# # Image Captioning with bottom-up and top-down attention
#
# In this example we'll caption an image with the pretrained model.

from skimage import io
import sys
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
# don't interpolate: show square pixels
plt.rcParams['image.interpolation'] = 'nearest'
# use grayscale output rather than a (potentially misleading) color heatmap
plt.rcParams['image.cmap'] = 'gray'

import cv2
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have not set the pythonpath.

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms

caffe.set_mode_gpu()
caffe.set_device(0)

caffe_root = '../'  # this file should be run from REPO_ROOT/scripts


# In[2]:


# Reduce the max number of region proposals, so that the bottom-up and top-down models can
# both fit on a 12GB gpu -> this may cause some demo captions to differ slightly from the
# generated outputs of ./experiments/caption_lstm/train.sh

# Previously 300 for evaluations reported in the paper
cfg['TEST']['RPN_POST_NMS_TOP_N'] = 105


# In[7]:


import os
import urllib

rcnn_weights = caffe_root + 'demo/resnet101_faster_rcnn_final.caffemodel'

# caption_weights = caffe_root + 'demo/lstm_iter_60000.caffemodel.h5' # cross-entropy trained
caption_weights = caffe_root + \
    'demo/lstm_scst_iter_1000.caffemodel.h5'  # self-critical trained

if os.path.isfile(rcnn_weights):
    print 'Faster R-CNN weights found.'
else:
    print 'Downloading Faster R-CNN weights...'
    url = "https://storage.googleapis.com/bottom-up-attention/resnet101_faster_rcnn_final.caffemodel"
    urllib.urlretrieve(url, rcnn_weights)

if os.path.isfile(caption_weights):
    print 'Caption weights found.'
else:
    print 'Downloading Caption weights...'
    url = "https://storage.googleapis.com/bottom-up-attention/%s" % caption_weights.split(
        '/')[-1]
    urllib.urlretrieve(url, caption_weights)


# ### 2. Visualization / feature extraction code for the bottom-up part

# In[8]:


# For visualization of bottom-up features

# Load classes
classes = ['__background__']
with open(caffe_root + 'demo/objects_vocab.txt') as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(caffe_root + 'demo/attributes_vocab.txt') as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())


# In[9]:


MIN_BOXES = 10
MAX_BOXES = 100

# Code for getting features from Faster R-CNN


def get_detections_from_im(net, cv2_im, image_id, conf_thresh=0.2):

    im = cv2_im
    scores, boxes, attr_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack(
            (cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(
            cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    objects = np.argmax(cls_prob[keep_boxes][:, 1:], axis=1)
    attrs = np.argmax(attr_prob[keep_boxes][:, 1:], axis=1)

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': cls_boxes[keep_boxes],
        'features': pool5[keep_boxes],
        'objects': objects,
        'attrs': attrs
    }


vocab_file = '../data/coco_splits/train_vocab.txt'

vocab = []
with open(vocab_file) as f:
    for word in f:
        vocab.append(word.strip())
print 'Loaded {:,} words into caption vocab'.format(len(vocab))


def translate(vocab, blob):
    caption = "";
    w = 0;
    while True:
        next_word = vocab[int(blob[w])]
        if w == 0:
            next_word = next_word.title()
        if w > 0 and next_word != "." and next_word != ",":
            caption += " ";
        if next_word == "\"" or next_word[0] == '"':
            caption += "\\";  # Escape
        caption += next_word;
        w += 1
        if caption[-1] == '.' or w == len(blob):
            break
    return caption


def lstm_inputs(dets):
    # Inputs to the caption network
    forward_kwargs = {'image_id': np.zeros((1, 3), np.float32)}
    forward_kwargs['image_id'][0, 1] = dets['image_h']
    forward_kwargs['image_id'][0, 2] = dets['image_w']

    forward_kwargs['num_boxes'] = np.ones((1, 1), np.float32)*dets['num_boxes']

    forward_kwargs['boxes'] = np.zeros((1, 101, 4), np.float32)
    forward_kwargs['boxes'][0, 1:dets['num_boxes']+1, :] = dets['boxes']

    forward_kwargs['features'] = np.zeros((1, 101, 2048), np.float32)
    forward_kwargs['features'][0, 0, :] = np.mean(dets['features'], axis=0)
    forward_kwargs['features'][0, 1:dets['num_boxes']+1, :] = dets['features']
    return forward_kwargs


feature_net = caffe.Net(
    caffe_root + 'demo/test.prototxt', rcnn_weights, caffe.TEST)


caption_net = caffe.Net(
    caffe_root + 'demo/decoder.prototxt', caption_weights, caffe.TEST)


video_capture = cv2.VideoCapture(0)
video_capture.set(3, 960)
video_capture.set(4, 640)

count = 0
cap = ''
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    count += 1
    ret, frame = video_capture.read()

    if not ret:
        print 'ayy'
        video_capture = cv2.VideoCapture(0)
        video_capture.set(3, 960)
        video_capture.set(4, 640)
        continue
    if count % 60 == 0:
        dets = get_detections_from_im(feature_net, frame, 0)
        
        # Process output detections
        forward_kwargs = lstm_inputs(dets)
        caption_net.forward(**forward_kwargs)
        image_ids = caption_net.blobs['image_id'].data
        captions = caption_net.blobs['caption'].data
        scores = caption_net.blobs['log_prob'].data     
        batch_size = image_ids.shape[0]

        cap = translate(vocab, captions[0])
        print cap

        
        batch_size = 1
        beam_size = 3
        print "Beam size: %d" % beam_size
        for n in range(batch_size):
            for b in range(beam_size):
                cap = translate(vocab, captions[n*beam_size+b])
                score = scores[n*beam_size+b]
                print '[%d] %.2f %s' % (b,score,cap)
        beam = 0
        cap = translate(vocab, captions[n*beam_size+beam])

    if cap:
        cv2.putText(frame,str(cap),(10,30), font, 1, (64,230,0), thickness=2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

