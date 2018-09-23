#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import tqdm
import cv2


def img_preprocess(images):

    new_imgs = []

    for img in images:

        img = np.reshape(img, (120, 160, 3))

        # transform the color image to grayscale
        img = cv2.cvtColor(img[:, :, :], cv2.COLOR_RGB2GRAY)

        # normalize image to range [0, 1] (divide each pixel by 255)
        # first transform the array of int to array of float else the division with 255 will return an array of 0s
        img = img.astype(float)
        img = img / 255

        # reshape the image to row vector [1, 120x160]
        img = np.reshape(img, (1, -1))

        new_imgs.append(img)

    new_imgs = np.array(new_imgs)

    return new_imgs


def load_data(file_path_img, file_path_vel):

    # read data
    images = np.load(file_path_img)
    velocities = np.load(file_path_vel)

    print('The dataset is loaded: {} images and {} velocities.'.format(images.shape[0], velocities.shape[0]))

    if not images.shape[0] == velocities.shape[0]:
        raise ValueError("The number of images and velocities must be the same.")

    # preprocess images
    images = img_preprocess(images)
    print images.shape

    # reshape images so that they are consistent with TensorFlow graph
    images = np.reshape(images, (-1, 120*160))

    # keep only the v, w velocities (1st column: v , 2nd column: w)
    velocities = velocities[:, [0, 1]]

    return images, velocities


def form_model_name(batch_size, lr, optimizer, epochs):
    return "batch={},lr={},optimizer={},epochs={}".format(batch_size, lr, optimizer, epochs)


class CNN_training:

    def __init__(self, batch, epochs, learning_rate, optimizer):

        self.batch_size = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer


    def backpropagation(self):

        # define the optimizer
        with tf.name_scope("Optimizer"):
            if self.optimizer == "Adam":
                return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer == "GDS":
                return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def loss_function(self):

        # define loss function and encapsulate its scope
        with tf.name_scope("Loss"):
            return tf.reduce_sum( tf.square(self.vel_pred - self.vel_true) )


    def model(self, x):

        with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

            # define the 4-d tensor expected by tensorflow
            # [-1: arbitrary num of images, img_height, img_width, num_channels]
            x_img = tf.reshape(x, [-1, 120, 160, 1])

            # define 1st convolutional layer
            conv_1 = tf.layers.conv2d(x_img, kernel_size=5, filters=4, padding="valid",
                                      activation=tf.nn.relu, name="conv_layer_1")

            max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2)

            # define 2nd convolutional layer
            conv_2 = tf.layers.conv2d(max_pool_1, kernel_size=5, filters=4, padding="valid",
                                      activation=tf.nn.relu, name="conv_layer_2")

            max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2)

            # define 3rd convolutional layer
            conv_3 = tf.layers.conv2d(max_pool_2, kernel_size=5, filters=8, padding="valid",
                                      activation=tf.nn.relu, name="conv_layer_3")

            max_pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=2, strides=2)

            # flatten tensor to connect it with the fully connected layers
            conv_flat = tf.layers.flatten(max_pool_3)

            # add 1st fully connected layers to the neural network
            fc_1 = tf.layers.dense(inputs=conv_flat, units=64, activation=tf.nn.relu, name="fc_layer_1")

            fc_1 = tf.nn.dropout(fc_1, keep_prob=0.5)

            # add 2nd fully connected layers to predict the driving commands
            fc_2 = tf.layers.dense(inputs=fc_1, units=2, name="fc_layer_2")

            return fc_2


    def find_batch_end(self, batch, data_size):

        if batch + self.batch_size > data_size - 1:
            return data_size
        else:
            return batch + self.batch_size


    def epoch_iteration(self, images, velocities, mode):

        epoch_loss = 0
        data_size = images.shape[0]

        if mode == 'train':

            for batch in tqdm.tqdm(range(0, data_size, self.batch_size), desc='Mini Batch Train'):
                batch_end = self.find_batch_end(batch, data_size)
                _, c = self.sess.run([self.opt, self.loss], feed_dict={self.x: images[batch:batch_end], self.vel_true: velocities[batch:batch_end]})
                epoch_loss += c

        elif mode == 'test':

            for batch in tqdm.tqdm(range(0, data_size, self.batch_size), desc='Mini Batch Test'):
                batch_end = self.find_batch_end(batch, data_size)
                c = self.sess.run(self.loss, feed_dict={self.x: images[batch:batch_end], self.vel_true: velocities[batch:batch_end]})
                epoch_loss += c

        avg_epoch_loss = epoch_loss/data_size

        return avg_epoch_loss


    def training(self, model_name, train_images, train_velocities, test_images, test_velocities):

        # define paths to save the TensorFlow logs
        model_path = os.path.join(os.getcwd(), 'tensorflow_logs', model_name)
        logs_train_path = os.path.join(model_path, 'train')
        logs_test_path = os.path.join(model_path, 'test')
        graph_path = os.path.join(model_path, 'graph')

        # manual scalar summaries for loss tracking
        man_loss_summary = tf.Summary()
        man_loss_summary.value.add(tag='Loss', simple_value=None)

        # define placeholder variable for input images (each images is a row vector [1, 19200 = 120x160x1])
        self.x = tf.placeholder(tf.float16, shape=[None, 120 * 160], name='x')

        # define placeholder for the true velocities
        # [None: tensor may hold arbitrary num of velocities, number of velocity predictions for each image]
        self.vel_true = tf.placeholder(tf.float16, shape=[None, 2], name="vel_true")
        self.vel_pred = self.model(self.x)

        self.loss = self.loss_function()
        self.opt = self.backpropagation()

        # initialize variables
        init = tf.global_variables_initializer()

        # Operation to save and restore all variables
        saver = tf.train.Saver()

        with tf.Session() as self.sess:

            # run initializer
            self.sess.run(init)

            # operation to write logs for Tensorboard
            tf_graph = self.sess.graph
            test_writer = tf.summary.FileWriter(logs_test_path, graph=tf.get_default_graph() )
            test_writer.add_graph(tf_graph)

            train_writer = tf.summary.FileWriter(logs_train_path, graph=tf.get_default_graph() )
            train_writer.add_graph(tf_graph)

            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pbtxt', as_text=True)
            tf.train.write_graph(tf_graph.as_graph_def(), graph_path, 'graph.pb', as_text=False)

            for epoch in range(self.epochs):

                # run train cycle
                avg_train_loss = self.epoch_iteration(train_images, train_velocities, 'train')

                # save the training loss using the manual summaries
                man_loss_summary.value[0].simple_value = avg_train_loss
                train_writer.add_summary(man_loss_summary, epoch)

                # run test cycle
                avg_test_loss = self.epoch_iteration(test_images, test_velocities, 'test')

                # save the test errors using the manual summaries
                man_loss_summary.value[0].simple_value = avg_test_loss
                test_writer.add_summary(man_loss_summary, epoch)

                # print train and test loss to monitor progress during training every 50 epochs
                if epoch % 50 == 0:
                    print("Epoch: {:04d} , train_loss = {:.6f} , test_loss = {:.6f}".format(epoch+1, avg_train_loss, avg_test_loss))

                # save weights every 100 epochs
                if epoch % 100 == 0:
                    saver.save(self.sess, logs_train_path, epoch)

        # close summary writer
        train_writer.close()
        test_writer.close()

