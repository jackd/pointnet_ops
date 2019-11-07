"""Compare pointnet_ops.group.group_point to tf.gather."""

import tensorflow as tf
from pointnet_ops import group
tf.compat.v1.enable_eager_execution()

batch_size = 7
n_dataset = 11
channel = 13
n_point = 17
n_sample = 19

points = tf.random.uniform((batch_size, n_dataset, channel))
idx = tf.random.uniform((batch_size, n_point, n_sample),
                        maxval=n_dataset,
                        dtype=tf.int32)

pointnet_impl = group.group_point(points, idx)
tf_impl = tf.gather(points, idx, batch_dims=1)

print(tf.reduce_max(tf.abs(pointnet_impl - tf_impl)).numpy())
