import numpy as np
from PIL import Image
import os
import caffe

import cv2

caffe.set_mode_gpu()
# load net
net = caffe.Net('deploy.prototxt', './snapshot/train_iter_25000.caffemodel', caffe.TEST)

for line in open('../data/card/ImageSets/Segmentation/val.txt'):
    path = os.path.join('../data/card/JPEGImages/', line.strip() + '.jpg')
    print path
    # label_path = os.path.join('../data/card/SegmentationClass/', line.strip() + '.png')
    # label = cv2.imread(label_path)
    # cv2.imwrite(line.strip() + '.png', label * 255)

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(path)
    im = im.resize((480, 320), Image.BILINEAR)
    im_orgin = np.array(im.copy())

    in_ = np.array(im, dtype = np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((106., 120., 114.))
    in_ = in_.transpose((2, 0, 1))

    # im = cv2.imread(path)
    # im = cv2.resize(im, (480, 320))
    # im_orgin = np.array(im.copy())
    # in_ = np.array(im, dtype = np.float32)
    # in_ -= np.array((106., 120., 114.))
    # in_ = in_.transpose((2, 0, 1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()
    # out = net.blobs['score_new'].data[0].argmax(axis = 0)
    out = net.blobs['prob'].data[0].argmax(axis = 0)

    # ind = 0
    # for x in net.blobs['prob'].data[0][0]:
    #     print 'ind', ind
    #     ind += 1
    #     print x

    # cv2.imwrite(os.path.join('results', line.strip() + '.png'), out * 255)


    out = np.array(out, dtype = np.uint8)
    im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im_orgin, [box], 0, (0, 0, 255), 2)


    cv2.imwrite(os.path.join('results', line.strip() + '.jpg'), im_orgin)
