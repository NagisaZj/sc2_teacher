import numpy as np
import tensorflow as tf
import tensorlayer as tl

from sc2_util import wrap
from sc2_util import FLAGS, flags
import teacher
import config_q
from absl import flags ,app

scr_pixels = 64
scr_num = 3
scope = 'learner'
regular = 0.05
lr = 1e-3
times_per_epoch = 100
epoches = 100
class learner:
    def __init__(self,scope,config,sess):
        self.config = config
        self.scope = scope
        with tf.variable_scope(scope) as scope:
            self.s = tf.placeholder(tf.float32,[None,scr_pixels,scr_pixels,scr_num],"state")
            self.action_16 = tf.placeholder(tf.float32,[None,scr_pixels*scr_pixels/16],"action_16")
            self.action_32 = tf.placeholder(tf.float32, [None, scr_pixels * scr_pixels / 4], "action_32")
            self.action_64 = tf.placeholder(tf.float32, [None, scr_pixels * scr_pixels ], "action_32")
            self.optimizer = tf.train.RMSPropOptimizer(lr, name='RMSProp')
            self._build_net()
            self.sess = sess
            tl.layers.initialize_global_variables(self.sess)




    def _build_net(self):
        regularizer = tf.contrib.layers.l2_regularizer(regular)
        with tf.variable_scope('var', regularizer=regularizer) as scope:
            self.map_64 = Util.block(self.s, self.config.bridge, "map_64")
        self.map_64_flat = tf.contrib.layers.flatten(self.map_64)
        self.prob_64 = tf.nn.softmax(self.map_64_float)
        self.loss_64 = -tf.reduce_sum(tf.multiply(self.prob_64,self.action_64))
        self.map_32 = tf.layers.average_pooling2D(self.map_64,[2,2],2,'same')
        self.map_32_flat = tf.contrib.layers.flatten(self.map_32)
        self.prob_32 = tf.nn.softmax(self.map_32_float)
        self.loss_32 = -tf.reduce_sum(tf.multiply(self.prob_32, self.action_32))
        self.map_16 = tf.layers.average_pooling2D(self.map_32, [2, 2], 2, 'same')
        self.map_16_flat = tf.contrib.layers.flatten(self.map_16)
        self.prob_16 = tf.nn.softmax(self.map_16_float)
        self.loss_16 = -tf.reduce_sum(tf.multiply(self.prob_16, self.action_16))

        self.opt_64 = self.optimizer.minimize(self.loss_64)
        self.opt_32 = self.optimizer.minimize(self.loss_32)
        self.opt_16 = self.optimizer.minimize(self.loss_16)
        self.params = tl.layers.get_variables_with_name(self.scope, True, False)

    def save_ckpt(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=self.sess, mode_name='model.ckpt', var_list=self.params,
                           save_dir=self.scope, printable=False)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=self.sess, var_list=self.params, save_dir=self.scope, printable=False)

    def train(self,state,action):
        loss = 0
        i = 0
        while i < 300:#loss > -0.9:
            _, loss = self.sess.run([self.opt,self.loss],feed_dict = {self.s:state,self.action:action})
            i = i+1
            if i % 30 == 0:
                print(loss)


    def train_16(self,state,action):

        for i in range(239):
            _, loss = self.sess.run([self.opt_16,self.loss_16],feed_dict = {self.s:[state[i]],self.action:[action[i]]})
            if i % 30 == 0:
                print(loss)

    def train_32(self,state,action):

        for i in range(239):
            _, loss = self.sess.run([self.opt_32,self.loss_32],feed_dict = {self.s:[state[i]],self.action:[action[i]]})
            if i % 30 == 0:
                print(loss)

    def train_64(self,state,action):

        for i in range(239):
            _, loss = self.sess.run([self.opt_64,self.loss_64],feed_dict = {self.s:[state[i]],self.action:[action[i]]})
            if i % 30 == 0:
                print(loss)


class generator:
    def __init__(self):
        self.env = wrap()

    def generate_expert(self):
        state, reward, done, info = self.env.reset()
        state_buffer, a16_buffer,a32_buffer,a64_buffer = [], [],[],[]
        while not done:
            a0, a1, a2 = teacher.action(state, info)
            action_64 = np.zeros((scr_pixels*scr_pixels,), dtype=np.float32)
            action_32 = np.zeros((scr_pixels * scr_pixels/4,), dtype=np.float32)
            action_16 = np.zeros((scr_pixels * scr_pixels/16,), dtype=np.float32)
            action_64[a1*scr_pixels+a2] = 1
            action_32[a1 * scr_pixels/4 + a2/2] = 1
            action_16[a1 * scr_pixels/16 + a2/4] = 1
            # print(action == 1)
            state_buffer.append([state])
            a16_buffer.append([action_16])
            a32_buffer.append([action_32])
            a64_buffer.append([action_64])

            state, reward, done, info = self.env.step(1 if a0 == 0 else int(2 + a1 * scr_pixels + a2))
        state_buffer, a16_buffer,a32_buffer,a64_buffer = np.vstack(state_buffer), np.vstack(a16_buffer), np.vstack(a32_buffer), np.vstack(a64_buffer)
        # print(state_buffer.shape)



        return state_buffer, a16_buffer,a32_buffer,a64_buffer



class Util:
    @staticmethod
    def block(x, config, name):
        with tf.variable_scope(name) as scope:
            layers = zip(config.types, config.filters, config.kernel_sizes,
                        config.strides, config.paddings, config.activations,
                        config.initializers)

            for type, filter, kernel_size, stride, padding, activation, initializer in layers:
                if type == 'conv':
                    x = tf.layers.conv2d(x,
                                         filters=filter,
                                         kernel_size=kernel_size,
                                         strides=stride,
                                         padding=padding,
                                         activation=activation,
                                         kernel_initializer=initializer)
                elif type == 'flat':
                    x = tf.contrib.layers.flatten(x)
                elif type == 'dense':
                    x = tf.layers.dense(x,
                                        filter,
                                        activation=activation,
                                        kernel_initializer=initializer)
            return x



def main(unused_argv):
    sess = tf.Session()
    learn = learner(scope,config_q.config,sess)
    gen = generator()
    learn.load_ckpt()
    for t in range(epoches):
        state,a16,a32,a64 = gen.generate_expert()
        learn.train_16(state,action16)
    learn.save_ckpt()


if __name__ =='__main__':
    app.run(main)



