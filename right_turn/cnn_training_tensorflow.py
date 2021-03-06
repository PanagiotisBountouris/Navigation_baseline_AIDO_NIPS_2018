#!/usr/bin/env python

from sklearn.model_selection import train_test_split
import time
import os
from cnn_training_functions import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():

    # define file paths for images and velocities
    file_path = os.path.join('..', '..', 'data', '15_09_2018')
    file_path_img = os.path.join(file_path, 'right_images.npy')
    file_path_vel = os.path.join(file_path, 'right_velocities.npy')

    # define batch_size (e.g 50, 100)
    batch_size = 100

    # define which optimizer you want to use (e.g "Adam", "GDS"). For "Adam" and "GDS" this script will take care the rest.
    # ATTENTION !! If you want to choose a different optimizer from these two, you will have to add it in the training functions.
    optimizer = "GDS"

    # define learning rate (e.g 1E-3, 1E-4, 1E-5):
    learning_rate = 1E-4

    # define total epochs (e.g 1000, 5000, 10000)
    epochs = 1000

    # read data
    print('Read data')
    images, velocities = load_data(file_path_img, file_path_vel)

    # split data to train and test sets
    print('Split data to train and test sets')

    img_train, img_test, vel_train, vel_test = train_test_split(images, velocities, test_size=0.1)
    print('Total data: {}, Train set: {} , Test set: {}.'.format(images.shape[0], img_train.shape[0], img_test.shape[0]))

    # construct model name based on the hyper parameters
    model_name = form_model_name(batch_size, learning_rate, optimizer, epochs)

    print('Starting training for {} model.'.format(model_name))

    # keep track of training time
    start_time = time.time()

    # train model
    cnn_train = CNN_training(batch_size, epochs, learning_rate, optimizer)
    cnn_train.training(model_name, img_train, vel_train, img_test, vel_test )

    # calculate total training time in minutes
    training_time = (time.time() - start_time) / 60

    print('Finished training of {} in {} minutes.'.format(model_name, training_time))

if __name__ == '__main__':
    main()
