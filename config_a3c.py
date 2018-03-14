#!/usr/bin/env python3
import tensorflow as tf


class Block:
    def __init__(self):
        self.types = None
        self.filters = None
        self.kernel_sizes = None
        self.strides = None
        self.paddings = None
        self.activations = None
        self.initializers = None


class config_a:
    xaiver = tf.contrib.layers.variance_scaling_initializer(factor=0.5, mode='FAN_AVG')
    relu = tf.nn.relu
    tanh = tf.nn.tanh
    softplus = tf.nn.softplus
    softmax = tf.nn.softmax
    sigmoid = tf.nn.sigmoid

    bridge = Block()
    bridge.types =          ['conv'] * 2 +['flat']+['dense']
    bridge.filters =        [16] + [32]  + [None] + [64]
    bridge.kernel_sizes =   [(8, 8)] + [(4,4)]+[None] + [None]
    bridge.strides =        [4]  +[2]+ [None] + [None]
    bridge.paddings =       ['SAME'] * 2+ [None] + [None]
    bridge.activations =    [relu] * 2+ [None] + [relu]
    bridge.initializers =   [xaiver]*2+ [None] + [xaiver]

    mu_1 = Block()
    mu_1.types = ['dense']
    mu_1.filters = [1]
    mu_1.kernel_sizes = [None]
    mu_1.strides = [None]
    mu_1.paddings = [None]
    mu_1.activations = [sigmoid]
    mu_1.initializers = [xaiver]

    mu_2 = Block()
    mu_2.types = ['dense']
    mu_2.filters = [1]
    mu_2.kernel_sizes = [None]
    mu_2.strides = [None]
    mu_2.paddings = [None]
    mu_2.activations = [sigmoid]
    mu_2.initializers = [xaiver]

    sigma_1 = Block()
    sigma_1.types = ['dense']
    sigma_1.filters = [1]
    sigma_1.kernel_sizes = [None]
    sigma_1.strides = [None]
    sigma_1.paddings = [None]
    sigma_1.activations = [softplus]
    sigma_1.initializers = [xaiver]

    sigma_2 = Block()
    sigma_2.types = ['dense']
    sigma_2.filters = [1]
    sigma_2.kernel_sizes = [None]
    sigma_2.strides = [None]
    sigma_2.paddings = [None]
    sigma_2.activations = [softplus]
    sigma_2.initializers = [xaiver]

    action = Block()
    action.types = ['dense']
    action.filters = [2]
    action.kernel_sizes = [None]
    action.strides = [None]
    action.paddings = [None]
    action.activations = [softmax]
    action.initializers = [xaiver]



class config_c:
    xaiver = tf.contrib.layers.variance_scaling_initializer(factor=0.5, mode='FAN_AVG')
    relu = tf.nn.relu
    tanh = tf.nn.tanh

    bridge = Block()
    bridge.types =          ['conv'] * 2
    bridge.filters =        [16]+ [32] 
    bridge.kernel_sizes =   [(8, 8)] + [(4, 4)] 
    bridge.strides =        [4]  + [2] 
    bridge.paddings =       ['SAME'] * 2
    bridge.activations =    [relu] * 2
    bridge.initializers =   [xaiver]*2



    value = Block()
    value.types = ['flat',  'dense', 'dense']
    value.filters = [None, 64,  1]
    value.kernel_sizes = [None] * 4
    value.strides = [None]*4
    value.paddings = [None]*4
    value.activations = [None] + [relu] + [None]
    value.initializers = [None] + [xaiver] * 2
