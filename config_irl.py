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
    bridge.filters =        [16] + [8]  +[1]
    bridge.kernel_sizes =   [(8, 8)] + [(4,4)]+[(1,1)]
    bridge.strides =        [2]  +[2]+ [1]
    bridge.paddings =       ['SAME'] * 4
    bridge.activations =    [relu]+[tanh] + [None]
    bridge.initializers =   [xaiver]*3

    reward_0 = Block()
    reward_0.types = ['flat']+['dense']
    reward_0.filters = [None]+[1]
    reward_0.kernel_sizes = [None]*2
    reward_0.strides = [None]*2
    reward_0.paddings = [None]*2
    reward_0.activations = [None]*2
    reward_0.initializers = [None]+[xaiver]