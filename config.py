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


class config:
    xaiver = tf.contrib.layers.variance_scaling_initializer(factor=0.5, mode='FAN_AVG')
    relu = tf.nn.relu
    tanh = tf.nn.tanh

    bridge = Block()
    bridge.types =          ['conv'] * 32
    bridge.filters =        [16] + [32] * 31
    bridge.kernel_sizes =   [(7, 7)] * 32
    bridge.strides =        [1] * 32
    bridge.paddings =       ['SAME'] * 32
    bridge.activations =    [tanh] * 32
    bridge.initializers =   [xaiver]*32

    map = Block()
    map.types =             ['conv'] * 32
    map.filters =           [32] * 30 + [16, 1]
    map.kernel_sizes =      [(3, 3)]*32
    map.strides =           [1] * 32
    map.paddings =          ['SAME'] * 32
    map.activations =       [tanh] * 31 + [None]
    map.initializers =      [xaiver] * 32

    other = Block()
    other.types = ['flat', 'dense', 'dense', 'dense']
    other.filters = [None, 64, 32, 2]
    other.kernel_sizes = [None] * 4
    other.strides = [None]*4
    other.paddings = [None]*4
    other.activations = [None] + [tanh]*2 + [None]
    other.initializers = [None] + [xaiver] * 3
