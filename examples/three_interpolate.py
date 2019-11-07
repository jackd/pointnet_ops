"""Compare pointnet_ops.interpolate.three_interpolate to tf.gather and sum."""

import tensorflow as tf
from pointnet_ops import interpolate
tf.compat.v1.enable_eager_execution()

batch_size = 7
m = 11
n = 13
c = 17

points = tf.random.uniform((batch_size, m, c), dtype=tf.float32)
idx = tf.random.uniform((batch_size, n, 3), maxval=m, dtype=tf.int32)
weight = tf.random.uniform((batch_size, n, 3), dtype=tf.float32)

pointnet_impl = interpolate.three_interpolate(points, idx, weight)


def get_tf_impl(points, idx, weight):
    gathered = tf.gather(points, idx, batch_dims=1)
    return tf.linalg.matvec(gathered, weight, transpose_a=True)
    # return tf.reduce_sum(gathered * tf.expand_dims(weight, axis=-1), axis=-2)


tf_impl = get_tf_impl(points, idx, weight)

print(tf.reduce_max(tf.abs(pointnet_impl - tf_impl)).numpy())
