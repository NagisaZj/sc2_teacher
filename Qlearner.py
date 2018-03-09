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
            self.action_64 = tf.placeholder(tf.float32,[None,scr_pixels*scr_pixels],"action_64")
            self.action_32 = tf.placeholder(tf.float32, [None, scr_pixels * scr_pixels/4], "action_32")
            self.action_16 = tf.placeholder(tf.float32, [None, scr_pixels * scr_pixels/16], "action_16")
            self.optimizer = tf.train.RMSPropOptimizer(lr, name='RMSProp')
            self._build_net()
            self.sess = sess
            tl.layers.initialize_global_variables(self.sess)




    def _build_net(self):
        regularizer = tf.contrib.layers.l2_regularizer(regular)
        with tf.variable_scope('var', regularizer=regularizer) as scope:
            self.map_64 = Util.block(self.s, self.config.bridge, "map_64")
        print(self.map_64.shape)
        self.map_32 = tf.layers.max_pooling2d(self.map_64,[2,2],2,'SAME')
        self.map_16 = tf.layers.max_pooling2d(self.map_32,[2,2],2,'SAME')
        self.flat_64 = tf.contrib.layers.flatten(self.map_64)
        self.flat_32 = tf.contrib.layers.flatten(self.map_32)
        self.flat_16 = tf.contrib.layers.flatten(self.map_16)
        self.prob_64 = tf.nn.softmax(self.flat_64)
        self.prob_32 = tf.nn.softmax(self.flat_32)
        self.prob_16 = tf.nn.softmax(self.flat_16)
        self.loss_64 = -tf.reduce_sum(tf.multiply(self.prob_64, self.action_64))
        self.loss_32 = -tf.reduce_sum(tf.multiply(self.prob_32, self.action_32))
        self.loss_16 = -tf.reduce_sum(tf.multiply(self.prob_16, self.action_16))
        self.opt64 = self.optimizer.minimize(self.loss_64)
        self.opt32 = self.optimizer.minimize(self.loss_32)
        self.opt16 = self.optimizer.minimize(self.loss_16)

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
            _, loss = self.sess.run([self.opt16,self.loss_16],feed_dict = {self.s:state,self.action_16:action})
            if i % 30 == 0:
                print(loss)

    def train_32(self,state,action):

        for i in range(239):
            _, loss = self.sess.run([self.opt32,self.loss_32],feed_dict = {self.s:[state[i]],self.action_32:[action[i]]})
            if i % 30 == 0:
                print(loss)

    def train_64(self,state,action):

        for i in range(239):
            _, loss = self.sess.run([self.opt64,self.loss_64],feed_dict = {self.s:[state[i]],self.action_64:[action[i]]})
            if i % 30 == 0:
                print(loss)



class generator:
    def __init__(self):
        self.env = wrap()

    def generate_expert(self):
        state, reward, done, info = self.env.reset()
        state_buffer, a64_buffer,a32_buffer,a16_buffer = [], [],[],[]
        while not done:
            a0, a1, a2 = teacher.action(state, info)
            action64 = np.zeros((scr_pixels*scr_pixels,), dtype=np.float32)
            action32 = np.zeros((scr_pixels * scr_pixels//4,), dtype=np.float32)
            action16 = np.zeros((scr_pixels * scr_pixels//16,), dtype=np.float32)
            action64[a1*scr_pixels+a2] = 1
            action32[a1 * scr_pixels//4 + a2//2] = 1
            action16[a1 * scr_pixels//16 + a2//4] = 1

            state_buffer.append([state])
            a64_buffer.append([action64])
            a32_buffer.append([action32])
            a16_buffer.append([action16])

            state, reward, done, info = self.env.step(1 if a0 == 0 else int(2 + a1 * scr_pixels + a2))
        state_buffer, a64_buffer,a32_buffer,a16_buffer = np.vstack(state_buffer), np.vstack(a64_buffer), np.vstack(a32_buffer), np.vstack(a16_buffer)
        # print(state_buffer.shape)



        return state_buffer, a64_buffer,a32_buffer,a16_buffer



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
        state,a64,a32,a16 = gen.generate_expert()
        learn.train_16(state,a16)
    learn.save_ckpt()


if __name__ =='__main__':
    app.run(main)



