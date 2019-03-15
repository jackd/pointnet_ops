''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

sample_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_sample_ops.so'))


def prob_sample(inp,inpr):
    '''
    input:
        batch_size * ncategory float32
        batch_size * npoints   float32
    returns:
        batch_size * npoints   int32
    '''
    return sample_ops.prob_sample(inp,inpr)


ops.NoGradient('ProbSample')
def gather_point(inp,idx):
    '''
    input:
        batch_size * ndataset * 3   float32
        batch_size * npoints        int32
    returns:
        batch_size * npoints * 3    float32
    '''
    return sample_ops.gather_point(inp,idx)


@ops.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sample_ops.gather_point_grad(inp,idx,out_g), None]

def farthest_point_sample(npoint,inp):
    '''
    input:
        int32
        batch_size * ndataset * 3   float32
    returns:
        batch_size * npoint         int32
    '''
    return sample_ops.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')
