import numpy as np
from PIL import Image
import os
import caffe

import cv2

caffe.set_mode_gpu()
caffe.set_device(1)
# load net
net = caffe.Net('deploy_cctn.prototxt', './snapshot/cctn_iter_600000.caffemodel', caffe.TEST)

for line in open('../data/synthtext/ImageSets/Segmentation/test.txt'):
    path = os.path.join('../data/synthtext/SynthText/', line.strip() + '.jpg')
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
    in_ -= np.array((0., 0., 0.))
    in_ = in_.transpose((2, 0, 1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take argmax for prediction
    net.forward()

    out = net.blobs['prob'].data[0].argmax(axis = 0)

    # prob = net.blobs['prob'].data[0]
    # _, height, width = prob.shape
    # out = np.zeros((height, width))
    # for h in xrange(height):
    #     for w in xrange(width):
    #         if prob[0][h][w] < prob[1][h][w] and prob[1][h][w] > 0.8:
    #             out[h][w] = 1


    # cv2.imwrite(os.path.join('results', line.replace('/', '').strip() + '.png'), out * 255)


    out = np.array(out, dtype = np.uint8)
    im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # cv2.drawContours(im_orgin, [contour], 0, (0, 0, 255), 2)

        # x, y, w, h = cv2.boundingRect(contour)
        # im_orgin = cv2.rectangle(im_orgin, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im_orgin, [box], 0, (0, 0, 255), 2)

        # epsilon = 0.1 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, epsilon, True)
        # cv2.drawContours(im_orgin, [approx], 0, (0, 0, 255), 2)

        # hull = cv2.convexHull(contour, returnPoints = True)
        # cv2.drawContours(im_orgin, [hull], 0, (0, 0, 255), 2)

        # ellipse = cv2.fitEllipse(contour)
        # im_orgin = cv2.ellipse(im_orgin, ellipse, (0, 255, 0), 2)

        # rows, cols = im_orgin.shape[:2]
        # [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((cols - x) * vy / vx) + y)
        # im_orgin = cv2.line(im_orgin, (cols - 1, righty), (0, lefty), (0, 0, 255), 2)
    cv2.imwrite(os.path.join('results', line.replace('/', '').strip() + '.jpg'), im_orgin)


# # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('./575fcd62N6f89c5a9.jpg')
# # im = im.resize((480, 320), Image.BILINEAR)
# im = im.resize((320, 480), Image.BILINEAR)
# im_orgin = np.array(im.copy())

# in_ = np.array(im, dtype = np.float32)
# in_ = in_[:,:,::-1]
# in_ -= np.array((0., 0., 0.))
# in_ = in_.transpose((2, 0, 1))

# # shape for input (data blob is N x C x H x W), set data
# net.blobs['data'].reshape(1, *in_.shape)
# net.blobs['data'].data[...] = in_

# # run net and take argmax for prediction
# net.forward()
# # out = net.blobs['score_new'].data[0].argmax(axis = 0)
# out = net.blobs['prob'].data[0].argmax(axis = 0)


# out = np.array(out, dtype = np.uint8)
# im2, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(im_orgin, [box], 0, (0, 0, 255), 2)


# cv2.imw
