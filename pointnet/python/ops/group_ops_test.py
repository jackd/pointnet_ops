from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from group_ops import query_ball_point, group_point


class GroupPointTest(tf.test.TestCase):

    def _test_grad(self, device):
        with tf.device(device):
            points = tf.constant(
                np.random.random((1,128,16)).astype(np.float32))
            xyz1 = tf.constant(np.random.random((1,128,3)).astype(np.float32))
            xyz2 = tf.constant(np.random.random((1,8,3)).astype(np.float32))
            radius = 0.3
            nsample = 32
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)

        with self.test_session():
            err = tf.test.compute_gradient_error(
                points, (1,128,16), grouped_points, (1,8,32,16))
        self.assertLess(err, 1e-4)

    def test_grad_gpu(self):
        self._test_grad('/gpu:0')

    def test_grad_cpu(self):
        self._test_grad('/cpu:0')

if __name__=='__main__':
  tf.test.main()
