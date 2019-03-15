# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for interpolate ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from interpolate_ops import interpolate
from interpolate_ops import nn
import tensorflow as tf


class GroupPointTest(tf.test.TestCase):

  def test_grad(self):
    with self.test_session():
      points = tf.constant(np.random.random((1,8,16)).astype('float32'))
      # print points
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      dist, idx = nn(xyz1, xyz2)
      weight = tf.ones_like(dist)/3.0
      interpolated_points = interpolate(points, idx, weight)
      # print interpolated_points
      err = tf.test.compute_gradient_error(
        points, (1,8,16), interpolated_points, (1,128,16))
      # print err
      self.assertLess(err, 1e-4)

if __name__=='__main__':
  tf.test.main()
