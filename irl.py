import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib as mpl
import tensorflow as tf
import config_irl
from sc2_util import wrap_clean
from sc2_util import FLAGS, flags
from absl import flags ,app
import teacher
import random_action
import tensorlayer as tl
import actor
import config_a3c

scr_pixels = 64
scr_num = 3
regular = 0.05
episodes = 200
steps = 20
lr = 1e-4
reward_decay = 0.9
GLOBAL_NET_SCOPE="Global_Net"

class reward_learner:

    def __init__(self,config):
        self.sess = tf.Session()
        self.s= tf.placeholder(tf.float32,shape=[None,scr_pixels,scr_pixels,scr_num],name ='state')
        self.action_expert = tf.placeholder(tf.float32,shape=[None,scr_pixels,scr_pixels],name ='action_expert')
        self.action_rand = tf.placeholder(tf.float32, shape=[None, scr_pixels, scr_pixels], name='action_random')
        self.config = config
        self.opt =  tf.train.RMSPropOptimizer(lr, name='RMSProp')
        self._build_net()
        with self.sess.as_default():
            tf.initialize_all_variables().run()



    def _build_net(self):
        regularizer = tf.contrib.layers.l2_regularizer(regular)
        with tf.variable_scope('var',regularizer=regularizer) as scope:
            self.map_raw = Util.block(self.s, self.config.bridge, "map")
        #self.reward_0 = Util.block(self.map,self.config.reward_0,"reward_0")
        self.map = tf.image.resize_images(self.map_raw,(scr_pixels,scr_pixels))
        self.map = tf.reshape(self.map,[-1,scr_pixels,scr_pixels])
        self.value_expert = tf.multiply(self.map,self.action_expert)
        self.value__expert = tf.reduce_sum(self.value_expert,axis = (1,2))
        self.value_rand = tf.multiply(self.map, self.action_rand)
        self.value_rand = tf.reduce_sum(self.value_rand, axis=(1, 2))
        self.equal = tf.multiply(self.action_expert,self.action_rand)
        self.equal = tf.reduce_sum(self.equal, axis=(1, 2))
        #self.value_max = tf.reduce_max(self.map,axis=(1,2))
        self.loss = tf.reduce_sum(self.value_rand + self.equal-self.value__expert) +  tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.opti = self.opt.minimize(self.loss)

        self.params = tl.layers.get_variables_with_name( 'var', True, False)


    def learn(self,state,action_expert,action_rand):
        for i in range(steps) :
            _,loss=self.sess.run([self.opti,self.loss],feed_dict={self.s:state,
                                                                  self.action_expert:action_expert,
                                                                  self.action_rand:action_rand})

            if i%10 ==0:            
                print(loss)
                if i == 0:
                    loss_init = loss

        return loss_init

    def save_ckpt(self):
        tl.files.exists_or_mkdir('var')
        tl.files.save_ckpt(sess=self.sess, mode_name='model.ckpt', var_list=self.params ,
                           save_dir='var', printable=False)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=self.sess, var_list=self.params, save_dir='var', printable=False)

    def reward(self,state,a0,a1,a2):
        action = np.zeros_like((scr_pixels,scr_pixels),dtype = np.float32)
        action[a1][a2] = 1
        if a0 ==0:
            return 0
        else:
            return self.sess.run([self.value__expert],feed_dict={self.s:[state],self.action_expert:[action]})[0]







class generator:
    def __init__(self,sess):
        self.env = wrap_clean()
        self.ac = actor.ACnet(GLOBAL_NET_SCOPE,None,config_a3c.config_a,config_a3c.config_c,sess)
        self.ac.load_ckpt()

    def generate_expert(self):
        state,reward,done,info = self.env.reset()
        state_buffer,a_buffer,info_buffer = [],[],[]
        while not done:
            a0,a1,a2 = teacher.action(state,info)
            action = np.zeros((scr_pixels,scr_pixels),dtype = np.float32)
            action[a1][a2]=1
            #print(action == 1)
            state_buffer.append([state])
            a_buffer.append([action])
            info_buffer.append([info])
            state,reward,done,info = self.env.step(1 if a0 == 0 else int(2 + a1 * scr_pixels + a2))
        state_buffer,a_buffer,info_buffer = np.vstack(state_buffer),np.vstack(a_buffer),np.vstack(info_buffer)
        #print(state_buffer.shape)
        buffer_v_target = []
        v_s_ = np.zeros_like(state_buffer[0])
        for s in state_buffer[::-1]:  # reverse buffer r
            v_s_ = s + reward_decay * v_s_  # compute v target
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()
        #self.env.reset()


        return buffer_v_target,a_buffer,info_buffer,state_buffer

    def generate_random(self,state,info):
        a_buffer = []
        for i in range(info.shape[0]):
            a0,a1,a2 = self.ac.choose_action([state[i]],[info[i]])
            action = np.zeros((scr_pixels, scr_pixels), dtype=np.float32)
            action[int(a1)][int(a2)] = 1
            #print(action == 1)
            a_buffer.append([action])
        a_buffer=  np.vstack(a_buffer)
        return a_buffer








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
                elif type == 'pool':
                    x = tf.layers.average_pooling2d(
                        x,
                        kernel_size,
                        stride,
                        padding
                    )
            return x



def main(unused_argv):
    sess = tf.Session()
    learner = reward_learner(config_irl.config)
    learner.load_ckpt()
    gen = generator(sess)
    loss = 100
    while(loss > -1):
        v,action_expert,info,state=gen.generate_expert()
        action_rand=gen.generate_random(state,info)
        loss = learner.learn(state,action_expert,action_rand)
    learner.save_ckpt()















if __name__ == '__main__':
    app.run(main)
