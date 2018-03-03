from __future__ import division, print_function, unicode_literals
from collections import deque
import time
import numpy as np
import os
import tensorflow as tf
import sys
import pdb

from sc2_util import wrap
from sc2_util import FLAGS, flags

flags.DEFINE_bool("test", False, "test (no learning and minimal epsilon)")
flags.DEFINE_integer("number_steps", 10000000, "total number of training steps")
flags.DEFINE_integer("explore_step", 12000, "total number of explorartion steps")
flags.DEFINE_integer("learn_freq", 4, "number of game steps between each training step")
flags.DEFINE_integer("save_steps",5000,"number of training steps between saving checkpoints")
flags.DEFINE_integer("training_start", 1000, "Game steps per agent step.")
flags.DEFINE_integer("mem_size", 10000, "Waiting for enough data")
flags.DEFINE_integer("batch_size", 64, "batch size for training")
flags.DEFINE_float("learning_rate", 5e-4, "learning rate for training")
flags.DEFINE_float("gamma", 0.999, "discount rate")
flags.DEFINE_string("save_dir", "D:/sc_dqn/sc2_sarsa/model", "directory for saving")

class Input:
    def __init__(self, model, name ="input"):
        state_shape = [None] + list(model.state_shape)

        with tf.variable_scope(name) as scope:
            self.state = tf.placeholder(tf.float32, shape=state_shape)
            self.next_state = tf.placeholder(tf.float32, shape=state_shape)
            self.done = tf.placeholder(tf.float32, shape=[None])
            self.reward = tf.placeholder(tf.float32, shape=[None])
            self.action = tf.placeholder(tf.int32, shape=[None])
            self.next_action = tf.placeholder(tf.int32, shape=[None])

            other_size, (x_size, _, _) = model.action_shape

            self.action_other = self.action
            self.action_map = self.action - other_size
            self.action_map_x = self.action_map % x_size
            self.action_map_y = self.action_map // x_size

            self.next_action_other = self.next_action
            self.next_action_map = self.next_action - other_size
            self.next_action_map_x = self.next_action_map % x_size
            self.next_action_map_y = self.next_action_map // x_size

    def infer_feed(self,state):
        return {self.state:[state]}

    def train_feed(self,state, action, reward, next_state,next_action, done):
        return {
            self.state:state,
            self.action:action,
            self.reward:reward,
            self.next_state:next_state,
            self.next_action:next_action,
            self.done:done
        }

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

class Train:
    def __init__(self, model, name):
        with tf.variable_scope(name) as scope:
            other_size, map_size = model.action_shape
            map_x = tf.one_hot(model.input.action_map_x, depth=map_size[0], axis=1)
            map_y = tf.one_hot(model.input.action_map_y, depth=map_size[1], axis=1)
            map_x = tf.expand_dims(map_x, 2)
            map_y = tf.expand_dims(map_y, 1)
            map_one_hot = tf.expand_dims(tf.matmul(map_x, map_y), 3)

            other_one_hot = tf.one_hot(model.input.action_other, depth=other_size, axis=1)

            next_map_x = tf.one_hot(model.input.next_action_map_x, depth=map_size[0], axis=1)
            next_map_y = tf.one_hot(model.input.next_action_map_y, depth=map_size[1], axis=1)
            next_map_x = tf.expand_dims(next_map_x, 2)
            next_map_y = tf.expand_dims(next_map_y, 1)
            next_map_one_hot = tf.expand_dims(tf.matmul(next_map_x, next_map_y), 3)

            next_other_one_hot = tf.one_hot(model.input.next_action_other, depth=other_size, axis=1)

            q_sa_map = tf.reduce_sum(model.q_map * map_one_hot, axis=[1, 2, 3])
            q_sa_other = tf.reduce_sum(model.q_other * other_one_hot, axis=1)
            q_sa = q_sa_map + q_sa_other

            next_q_sa_map = tf.reduce_sum(model.t_q_map * next_map_one_hot, axis=[1, 2, 3])
            next_q_sa_other = tf.reduce_sum(model.t_q_other * next_other_one_hot, axis=1)
            next_q_sa = next_q_sa_map + next_q_sa_other

            self.loss = model.input.reward + (1.0 - model.input.done) * model.gamma * tf.stop_gradient(next_q_sa) - q_sa
            optimizer = tf.train.GradientDescentOptimizer(model.learning_rate)
            self.train = optimizer.minimize(self.loss)

class ModelVar:
    def __init__(self, save_dir, model):
        self.save_dir = save_dir
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.name)
        self.init = tf.variables_initializer(self.vars)
        self.saver = tf.train.Saver(var_list=self.weights, max_to_keep=5)

    def save(self, sess, step):
        path = os.path.join(self.save_dir, 'model')
        self.saver.save(sess, path, global_step=step)

    def restore(self, sess, step=None):
        if step is None:
            path = tf.train.latest_checkpoint(self.save_dir)
            print('restore from %s' % path)
            if path is None:
                return False
        else:
            path = os.path.join(self.save_dir, 'model-%d' % step)
        self.saver.restore(sess, path)

        return True


class Model:
    def __init__(self, config, state_shape, action_shape, lr, gamma, save_dir, name='dqn'):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.config = config
        self.name = name
        self.learning_rate = lr
        self.gamma = gamma
        self.save_dir = save_dir

        with tf.variable_scope(name) as scope:
            self.input = Input(self, name)#?

            with tf.variable_scope('step') as scope:
                self.global_step = tf.Variable(0, trainable = False, name='global_step')
                self.explicit_step = tf.assign_add(self.global_step, 1)

            with tf.variable_scope('forward') as scope:
                self.bridge = Util.block(self.input.state, config.bridge, 'bridge')
                self.q_map = Util.block(self.bridge, config.map, 'map')
                self.q_other = Util.block(self.bridge, config.other, 'other')

            with tf.variable_scope('forward', reuse=True) as scope:
                self.t_bridge = Util.block(self.input.next_state, config.bridge, 'bridge')
                self.t_q_map = Util.block(self.t_bridge, config.map, 'map')
                self.t_q_other = Util.block(self.t_bridge, config.other, 'other')
            self.train = Train(self, 'train')
            self.var = ModelVar(self.save_dir, self)

        self.sess = tf.Session()
        self.sess.run(self.var.init)
        self.var.restore(self.sess)


    def get_step(self):
        return self.sess.run(self.global_step)

    def start_infer(self, state):
        return self.sess.run(
            (self.q_map, self.q_other, self.explicit_step),
            self.input.infer_feed(state))

    def predict_infer(self, state):
        return self.sess.run(
            (self.q_map, self.q_other, self.global_step),
            self.input.infer_feed(state))

    def start_train(self, inputs):
        _, loss, step = self.sess.run(
            [self.train.train, self.train.loss, self.global_step],
            self.input.train_feed(*inputs))
        return loss, step

    def save(self, step):
        self.var.save(self.sess, step)

class Explorer:
    def __init__(self,
            explore_step, eps_min, eps_max,
            map_x, map_y,
            other_size, random_other_rate):
        self.explore_step = explore_step
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.map = (map_x, map_y)
        self.other = other_size
        self.total = map_x * map_y + other_size

        self.random_other_rate = random_other_rate

    def make_action(self, step, q_map, q_other, test=False):
        eps = self.eps_max - (self.eps_max - self.eps_min) * step / self.explore_step
        eps = max(self.eps_min, eps)

        if (not test) and (np.random.rand() < eps): # explore\
            print("explore")
            if np.random.rand() < self.random_other_rate:
                return np.random.randint(self.other)
            else:
                return self.other + np.random.randint(self.total - self.other)
        else: # exploit
            # XXX the correctness of code is relative to the row & column order of np.array
            q_map.shape = (self.map[0] * self.map[1],)
            q_other = q_other.reshape(self.other,)
            pm = np.argmax(q_map)
            po = np.argmax(q_other)

            if q_map[pm] > q_other[po]:
                #q_other.shape = (1,self.other)
                return pm + self.other
            else:
                #q_other.shape = (1, self.other)
                return po

def main():
    global env

    env = wrap()
    state_shape = env.state_shape()
    action_shape = env.action_shape()
    from config import config

    dqn = Model(config, state_shape, action_shape,
                FLAGS.learning_rate, FLAGS.gamma, FLAGS.save_dir)
    explorer = Explorer(FLAGS.explore_step,
                        0.01, 1.0, action_shape[1][0], action_shape[1][1],
                        action_shape[0], 0.2)

    done = True
    step = dqn.get_step()

    while step < FLAGS.number_steps:
        if done:
            state, _, _, info = env.reset()
            act_rem = None
            #print(state.shape)
        q_map, q_other, step = dqn.start_infer(state)
        action = act_rem if act_rem != None else explorer.make_action(step, q_map, q_other, FLAGS.test)

        next_state, reward, done, info = env.step(action)
        next_q_map, next_q_other, step = dqn.predict_infer(state)
        act_rem = explorer.make_action(step, next_q_map, next_q_other, FLAGS.test)

        inputs = [[state],[action],[reward],[next_state],[act_rem],[done]]

        loss, step = dqn.start_train(inputs)
        print(q_other.shape)
        print('q_max: %f, q_min: %f, q_noop: %f, q_select: %f, action: %d, reward: %f' % (
        max(q_map.max(), q_other.max()), min(q_map.min(), q_other.min()), q_other[0][0], q_other[0][1], action, reward))
        print('step: %d, loss: %f' % (step, loss))

        if step % FLAGS.save_steps == 0:
            print('')
            print('model saved. step: %d' % step)
            dqn.save(step)

if __name__ == "__main__":
        main()

