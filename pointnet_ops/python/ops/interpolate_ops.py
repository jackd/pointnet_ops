"""Use group ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

interpolate_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_interpolate_ops.so'))


def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_ops.three_nn(xyz1, xyz2)


ops.NoGradient('ThreeNN')


def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_ops.three_interpolate(points, idx, weight)


@ops.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [
        interpolate_ops.three_interpolate_grad(points, idx, weight, grad_out),
        None, None
    ]
