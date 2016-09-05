#! /bin/sh
#
# run.sh
# Copyright (C) 2016 qingze <qingze@localhost.localdomain>
#
# Distributed under terms of the MIT license.
#

#export PYTHONPATH=/export/wangqingze/caffe/python:../

#/export/wangqingze/caffe/build/tools/caffe test --model ./val.prototxt\
#                                                --weights ./fcn8s-heavy-pascal.caffemodel \
#                                                --gpu 2 \
#                                                --iterations 500


export PYTHONPATH=/export/wangqingze/software/caffe/python:../
/export/wangqingze/software/caffe/build/tools/caffe train -solver ./solver.prototxt \
                                                -weights ../card-fcn8s/VGG_CNN_M_1024.v2.caffemodel \
                                                -gpu 2 \

