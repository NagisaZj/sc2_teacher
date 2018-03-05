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
            self.action = tf.placeholder(tf.float32,[None,scr_pixels*scr_pixels],"action")
            self.optimizer = tf.train.RMSPropOptimizer(lr, name='RMSProp')
            self._build_net()
            self.sess = sess
            tl.layers.initialize_global_variables(self.sess)




    def _build_net(self):
        regularizer = tf.contrib.layers.l2_regularizer(regular)
        with tf.variable_scope('var', regularizer=regularizer) as scope:
            self.map_raw = Util.block(self.s, self.config.bridge, "map")
        # self.reward_0 = Util.block(self.map,self.config.reward_0,"reward_0")
        self.map = tf.image.resize_images(self.map_raw, (scr_pixels, scr_pixels))
        self.map = tf.reshape(self.map, [-1, scr_pixels, scr_pixels])
        self.flat = tf.contrib.layers.flatten(self.map)
        self.q_max = tf.reduce_max(self.map,axis = (1,2))
        self.prob = tf.nn.softmax(self.flat)
        self.loss = -tf.reduce_sum(tf.multiply(self.prob,self.action)) #+ tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.opt = self.optimizer.minimize(self.loss)
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



class generator:
    def __init__(self):
        self.env = wrap()

    def generate_expert(self):
        state, reward, done, info = self.env.reset()
        state_buffer, a_buffer = [], []
        while not done:
            a0, a1, a2 = teacher.action(state, info)
            action = np.zeros((scr_pixels*scr_pixels,), dtype=np.float32)
            action[a1*scr_pixels+a2] = 1
            # print(action == 1)
            state_buffer.append([state])
            a_buffer.append([action])

            state, reward, done, info = self.env.step(1 if a0 == 0 else int(2 + a1 * scr_pixels + a2))
        state_buffer, a_buffer = np.vstack(state_buffer), np.vstack(a_buffer)
        # print(state_buffer.shape)



        return state_buffer, a_buffer



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
        state,action = gen.generate_expert()
        learn.train(state,action)
    learn.save_ckpt()


if __name__ =='__main__':
    app.run(main)



