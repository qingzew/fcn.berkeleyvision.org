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
#                                                -weights /export/wangqingze/caffe/examples/vgg_ocr/snapshot/vgg_all_from_scratch_iter_30000.caffemodel \
#                                                -gpu 3 \

#/export/wangqingze/software/caffe/build/tools/caffe train -solver ./solver_cctn.prototxt \
#                                                -weights /export/wangqingze/caffe/examples/vgg_ocr/snapshot/vgg_parallel_iter_20000.caffemodel \
#                                                -gpu 2 \

#/export/wangqingze/software/caffe/build/tools/caffe train -solver ./solver_bilinear.prototxt \
#                                                -weights ./snapshot/shrink_iter_120000.caffemodel \
#                                                -gpu 3 \


python infer.py
