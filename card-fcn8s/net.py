import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split):
    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(106., 120., 114.),
            seed=1337)

    pydata_params['voc_dir'] = '../data/card'
    pylayer = 'VOCSegDataLayer'

    # if split == 'train':
    #     pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
    #     pylayer = 'SBDDSegDataLayer'
    # else:
    #     pydata_params['voc_dir'] = '../data/pascal/VOCdevkit/VOC2012'
    #     pylayer = 'VOCSegDataLayer'

    n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6_conv, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7_conv, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score_fr_new = L.Convolution(n.drop7, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.upscore2_new = L.Deconvolution(n.score_fr_new,
        convolution_param=dict(num_output=2, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=1)])

    n.score_pool4_new = L.Convolution(n.pool4, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.score_pool4c_new = crop(n.score_pool4_new, n.upscore2_new)

    n.fuse_pool4_new = L.Eltwise(n.upscore2_new, n.score_pool4c_new,
            operation=P.Eltwise.SUM)

    n.upscore_pool4_new = L.Deconvolution(n.fuse_pool4_new,
        convolution_param=dict(num_output=2, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=1)])

    n.score_pool3_new = L.Convolution(n.pool3, num_output=2, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    n.score_pool3c_new = crop(n.score_pool3_new, n.upscore_pool4_new)

    n.fuse_pool3_new = L.Eltwise(n.upscore_pool4_new, n.score_pool3c_new,
            operation=P.Eltwise.SUM)

    n.upscore8_new = L.Deconvolution(n.fuse_pool3_new,
        convolution_param=dict(num_output=2, kernel_size=16, stride=8,
            bias_term=False),
        param=[dict(lr_mult=1)])

    n.score_new = crop(n.upscore8_new, n.data)

    n.loss = L.SoftmaxWithLoss(n.score_new, n.label,
            # loss_param=dict(normalize=False, ignore_label=255))
            loss_param=dict(normalize=True))

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('val')))

if __name__ == '__main__':
    make_net()
