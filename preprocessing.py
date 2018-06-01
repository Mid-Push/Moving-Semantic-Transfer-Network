
import logging
import numpy as np
import tensorflow as tf


models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        # set default parameters
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

def preprocessing(inputs, model_fn):
    inputs = tf.cast(inputs, tf.float32)
    channels = int(inputs.get_shape()[-1])
    if channels == 1 and model_fn.num_channels == 3:
        print 'really GREY TO RGB?'
        logging.info('Converting grayscale images to RGB')
        inputs = tf.image.grayscale_to_rgb(inputs)
    elif channels == 3 and model_fn.num_channels == 1:
        print 'really RGB TO GREY?'
	inputs = tf.image.rgb_to_grayscale((inputs))
    if model_fn.range is not None:
        print 'range=[255]?'
	inputs = model_fn.range * inputs
    if model_fn.default_image_size is not None:
        size = model_fn.default_image_size
        logging.info('Resizing images to [{}, {}]'.format(size, size))
        inputs = tf.image.resize_images(inputs, [size, size])
	print 'after resize ',inputs.get_shape()
    if model_fn.mean is not None:
        logging.info('Performing mean subtraction.')
        inputs = inputs - tf.reshape(tf.constant(model_fn.mean),[-1,3])
	print 'after mean ',inputs.get_shape()
    if model_fn.bgr:
        logging.info('Performing BGR transposition.')
        inputs = inputs[:, :, [2, 1, 0]]
    #print 'start nomrliazation (x-mean)/sqrt(var+epsilon)'
    #mean,var=tf.nn.moments(inputs,axes=[0])
    #inputs=(inputs-mean)/tf.sqrt(var+1e-8)
    return inputs

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

def rgb2gray(image):
    return tf.reduce_sum(tf.multiply(image, tf.constant(RGB2GRAY)),
                         -1,
                         keep_dims=True)

def gray2rgb(image):
    return tf.multiply(image, tf.constant(RGB2GRAY))
