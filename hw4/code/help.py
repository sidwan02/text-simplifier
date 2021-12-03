import tensorflow as tf

a = tf.convert_to_tensor([[1, 2, 3, 4, 1, 2, 1], [1, 1, 2, 3, 2, 3, 2]])

ans = tf.where(a == 1, 0, 1)

print(ans)

print(tf.math.reduce_sum(ans))