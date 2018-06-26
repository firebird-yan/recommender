import tensorflow as tf
import numpy as np

x = np.random.rand(100)
y = 0.3 * x + 0.1

weight = tf.Variable(tf.random_uniform([1]))
bias = tf.Variable(tf.zeros([1]))
y_hat = tf.add(tf.multiply(weight, x), bias)
loss = tf.reduce_mean(tf.square(y - y_hat))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(100):
        _, _loss, _weight, _bias = sess.run([train, loss, weight, bias])
        print('---->', _weight, _bias, _loss)
