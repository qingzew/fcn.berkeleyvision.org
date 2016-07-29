#! /bin/sh
#
# run.sh
# Copyright (C) 2016 qingze <qingze@localhost.localdomain>
#
# Distributed under terms of the MIT license.
#

#export PYTHONPATH=/export/wangqingze/caffe/python:../
export PYTHONPATH=/export/wangqingze/software/caffe/python:../

#python net.py

#/export/wangqingze/software/caffe/build/tools/caffe train -solver ./solver_bilinear.prototxt \
#                                                -weights ./VGG_CNN_M_1024.v2.caffemodel \
#                                                -gpu 2 \

#/export/wangqingze/software/caffe/build/tools/caffe train -solver ./solver_bilinear.prototxt \
#                                                -weights /export/wangqingze/caffe/examples/vgg_ocr/snapshot/vgg_all_from_scratch_iter_30000.caffemodel \
#                                                -gpu 1 \


#/export/wangqingze/software/caffe/build/tools/caffe test -model ./train_bilinear.prototxt \
#                                                -weights ./snapshot/train_lr6_iter_35000.caffemodel \
#                                                -gpu 1 \
#                                                -iterations 50


python infer.py
