import tensorflow as tf

input1 = tf.constant(1)
input2 = tf.constant(2)
input3 = tf.constant(3)

add = tf.add(input1, input2)
mult = tf.multiply(add, input3)

with tf.Session() as sess:
    print(sess.run(mult))

