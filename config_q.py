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
    softplus = tf.nn.softplus
    softmax = tf.nn.softmax
    sigmoid = tf.nn.sigmoid

    bridge = Block()

    bridge.types =          ['conv'] * 3
    bridge.filters =        [16] + [16]  +[1]
    bridge.kernel_sizes =   [(8, 8)] + [(4,4)]+[(1,1)]
    bridge.strides =        [1]  +[1]+ [1]
    bridge.paddings =       ['SAME'] * 3
    bridge.activations =    [relu]+[tanh] + [None]
    bridge.initializers =   [xaiver]*3

    '''bridge.types =          ['conv'] * 10
    bridge.filters =        [16] + [16]*8  +[1]
    bridge.kernel_sizes =   [(8, 8)] + [(4,4)]*8+[(1,1)]
    bridge.strides =        [2]  +[2]+ [1]*8
    bridge.paddings =       ['SAME'] * 10
    bridge.activations =    [relu]*8+[tanh] + [None]
    bridge.initializers =   [xaiver]*10'''

