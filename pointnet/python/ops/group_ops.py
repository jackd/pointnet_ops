"""Use group ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
from tensorflow.python import array_ops
from tensorflow.python import math_ops

group_ops = load_library.load_op_library(
   resource_loader.get_path_to_datafile('_group_ops.so'))


def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in
            each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return group_ops.query_ball_point(xyz1, xyz2, radius, nsample)

ops.NoGradient('QueryBallPoint')


def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset
            points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return group_ops.selection_sort(dist, k)

ops.NoGradient('SelectionSort')


def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample
            from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values
            sampled from points
    '''
    return group_ops.group_point(points, idx)
@ops.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [group_ops.group_point_grad(points, idx, grad_out), None]


def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.shape[0].value
    n = xyz1.shape[1].value
    c = xyz1.shape[2].value
    m = xyz2.shape[1].value
    xyz1 = array_ops.reshape(xyz1, (b,1,n,c))
    xyz2 = array_ops.reshape(xyz2, (b,m,1,c))
    dist = math_ops.reduce_sum(math_ops.squared_distance(xyz1, xyz2), -1)
    outi, out = select_top_k(k, dist)
    idx = array_ops.slice(outi, [0,0,0], [-1,-1,k])
    val = array_ops.slice(out, [0,0,0], [-1,-1,k])
    return val, idx
