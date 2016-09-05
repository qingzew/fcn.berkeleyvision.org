import caffe

import numpy as np
from PIL import Image

import random
import cv2

class SynSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.syn_dir = params['syn_dir']
        self.bbox_npy = params['bbox_npy']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.syn_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

        self.bboxes = np.load(self.bbox_npy, encoding = 'latin1').item()

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/SynthText/{}.jpg'.format(self.syn_dir, idx))
        im = im.resize((420, 320), Image.BILINEAR)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = cv2.imread('{}/SynthText/{}.jpg'.format(self.syn_dir, idx))
        label = np.zeros((im.shape[0], im.shape[1]))

        boxes = self.bboxes[idx + '.jpg']

        for n in xrange(boxes.shape[-1]):
            # box = list(map(list, boxes[:, :, n].transpose()))
            # box = np.int0(box)
            # cv2.drawContours(label, [box], 0, 1, -1)

            box = list(map(list, boxes[:, :, n].transpose()))
            rect = cv2.minAreaRect(np.array(box))

            if rect[1][0] / (rect[1][1] + 1e-5) > 1.:
                rect = (rect[0], (rect[1][0], rect[1][1] * .4), rect[2])
            else:
                rect = (rect[0], (rect[1][0] * .4, rect[1][1]), rect[2])

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(label, [box], 0, 1, -1)



        label = cv2.resize(label, (420, 320), cv2.INTER_LINEAR)

        label = np.array(label, dtype=np.uint8)
        label[label > 0] = 1
        # print 'saving'
        # cv2.imwrite(str(idx).replace('/', '') + '.png', label * 255)
        label = label[np.newaxis, ...]
        return label

