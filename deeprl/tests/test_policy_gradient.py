import unittest
import os

import numpy as np
import tensorflow as tf

import deeprl.policy_gradient.train_pg as pg

class TestPolicyGradient(tf.test.TestCase):
    def test_mlp(self):
        scope = "test-mlp"
        input_size = 10
        output_size = 5
        hidden_layers = [100, 100]
        x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        y = pg.build_mlp(input_placeholder=x,
                         output_size=output_size,
                         scope=scope,
                         hidden_layers=hidden_layers)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Check internal dimensions.
            print("Checking internal weight sizes.")
            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope+"/dense_1"))
            #print(tf.get_variable(scope+"/dense/kernel"))

            with tf.variable_scope(scope+"/dense", reuse=True):
                self.assertTrue(tf.get_variable("kernel").shape==(input_size, hidden_layers[0]))
            with tf.variable_scope(scope+"/dense_1", reuse=True):
                self.assertTrue(tf.get_variable("kernel").shape==(hidden_layers[0], hidden_layers[1]))
            with tf.variable_scope(scope+"/dense_2", reuse=True):
                self.assertTrue(tf.get_variable("kernel").shape==(hidden_layers[1], output_size))

            num_samples = 5
            x_sample = np.ones((num_samples, 10))
            y_model = sess.run(y, feed_dict={x: x_sample})
            self.assertTrue(y_model.shape==(num_samples, output_size))
            #self.assertTrue(False)

if __name__ == "__main__":
    tf.test.main()
