import multiprocessing, threading, os, shutil
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import pysc2
from pysc2 import agents, env
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2 import lib
from pysc2.env import environment
from absl import flags, app

from sc2_util import wrap
from sc2_util import FLAGS, flags
import teacher
import matplotlib.pyplot as plt

supervise = 10.0
MAX_GLOBAL_EP =20000 
GLOBAL_NET_SCOPE = "Global_Net"
UPDATE_GLOBAL_ITER = 40
scr_pixels = 64
scr_num = 5
scr_bound = [0, scr_pixels - 1]
entropy_gamma = -5
steps = 40
action_speed = 8
reward_discount = GAMMA = 0.9
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
N_WORKERS = 64 
N_A = 2
available_len = 524
available_len_used = 2
save_path = "/models"
game = ["CollectMineralShards_2","CollectMineralShards_5","CollectMineralShards_10","CollectMineralShards_15","CollectMineralShards_20",]
score_high = [6,15,25,35,1000]
score_low = [-100,5,10,15,20]
#sigma_pow = 0.10
class ACnet:
    def __init__(self, scope, globalAC=None,  config_a=None, config_c=None):
        self.scope = scope
        self.config_a = config_a
        self.config_c = config_c
        if scope == GLOBAL_NET_SCOPE:  # build global net
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, scr_pixels, scr_pixels, scr_num], "S")
                self.available = tf.placeholder(tf.float32, [None, available_len_used], "available_actions")
                self._build_net()

                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)

                with tf.name_scope("choose_a"):  # choose actions,do not include a0 as a0 is discrete
                    mu_1, sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                    mu_2, sigma_2 = self.mu_2 * scr_bound[1], self.sigma_2 + 1e-5
                    self.a_1 = tf.clip_by_value(
                        tf.squeeze(tf.contrib.distributions.Normal(mu_1, sigma_1).sample(1), axis=0), *scr_bound)
                    self.a_2 = tf.clip_by_value(
                        tf.squeeze(tf.contrib.distributions.Normal(mu_2, sigma_2).sample(1), axis=0), *scr_bound)

        else:
            with tf.variable_scope(scope):  # else, build local network
                self.s = tf.placeholder(tf.float32, [None, scr_pixels, scr_pixels, scr_num], "S")
                self.available = tf.placeholder(tf.float32, [None, available_len_used], "available_actions")
                self.a0 = tf.placeholder(tf.int32, [None, 1], "a0")
                self.a1 = tf.placeholder(tf.float32, [None, 1], "a1")
                self.a2 = tf.placeholder(tf.float32, [None, 1], "a2")
                self.a0_exp = tf.placeholder(tf.int32, [None, 1], "a0_exp")
                self.a1_exp = tf.placeholder(tf.float32, [None, 1], "a1_exp")
                self.a2_exp = tf.placeholder(tf.float32, [None, 1], "a2_exp")
                '''self.a0_ex = tf.placeholder(tf.int32, [None, 1], "a0_ex")
                self.a1_ex = tf.placeholder(tf.float32, [None, 1], "a1_ex")
                self.a2_ex = tf.placeholder(tf.float32, [None, 1], "a2_ex")'''
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self._build_net()

                td = tf.subtract(self.v_target, self.value, name='TD_error')
                self.td = td
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

            with tf.name_scope('wrap_a_out'):
                self.test = self.sigma_1[0]
                mu_1, sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                mu_2, sigma_2 = self.mu_2 * scr_bound[1], self.sigma_2 + 1e-5

            normal_dist_1 = tf.contrib.distributions.Normal(mu_1, sigma_1)
            normal_dist_2 = tf.contrib.distributions.Normal(mu_2, sigma_2)

            with tf.name_scope("a_loss"):  # build loss function
                #self.sigma_loss = tf.reduce_mean(tf.square(self.sigma_1)+tf.square(self.sigma_2))
                log_prob0 = tf.reduce_sum(tf.log(self.action) * tf.one_hot(self.a0, N_A, dtype=tf.float32), axis=1,
                                          keep_dims=True)
                log_prob1 = normal_dist_1.log_prob(self.a1)
                log_prob2 = normal_dist_2.log_prob(self.a2)
                log_prob = tf.zeros_like(log_prob0)
                self.loss_exp = normal_dist_1.log_prob(self.a1_exp)+normal_dist_2.log_prob(self.a2_exp)
                print(self.a0.shape)
                '''
                for i in range(self.a0.shape[0]):
                    if self.a0[i,0]!=0:
                        log_prob[i,0]=log_prob0[i,0]+log_prob1[i,0]+log_prob2[i,0]
                    else:
                        log_prob[i,0]=log_prob0[i,0]
                '''
                log_prob = log_prob0 + log_prob1 + log_prob2 

                exp_v = log_prob * td

                entropy0 = -tf.reduce_sum(self.action * tf.log(self.action + 1e-5),
                                          axis=1, keep_dims=True)
                entropy1 = normal_dist_1.entropy()
                entropy2 = normal_dist_2.entropy()
                '''
                for i in range(self.a0.shape[0]):
                    if self.a0[i,0]!=0:
                        entropy[i,0] = entropy0[i,0] + entropy1[i,0] + entropy2[i,0]
                    else:
                        entropy[i, 0] = entropy0[i, 0]
                '''
                entropy = entropy1 + entropy2  # add entropy to encourage exploration
                # entropy = tf.zeros_like(entropy)
                # TODO: action a0(select all) and action a1(move_screen) should have different entropy and loss,
                # TODO: as the number of parameters are different(1 for a0, and 3 for a1) HOW TO IMPLEMENT?

                self.entropy = entropy
                self.exp_v = entropy * entropy_gamma + exp_v  +self.loss_exp * supervise
                self.a_loss = tf.reduce_mean(-self.exp_v) #+ self.sigma_loss * sigma_pow
                self.exp_loss = tf.reduce_mean(self.loss_exp)

            with tf.name_scope('choose_a'):  # use local params to choose action
                self.a_1 = tf.clip_by_value(tf.squeeze(normal_dist_1.sample(1), axis=0), *scr_bound)
                self.a_2 = tf.clip_by_value(tf.squeeze(normal_dist_2.sample(1), axis=0), *scr_bound)

            with tf.name_scope('local_grad'):
                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
        tl.layers.initialize_global_variables(sess)

    def update_global_high(self, feed_dict):  # run by a local
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test],
                           feed_dict)  # local grads applies to global net
        return t

    def update_global_low(self, feed_dict):
        sess.run([self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, avail_new):  # run by a local
        prob_weights = sess.run(self.action, feed_dict={self.s: s,
                                                        self.available: avail_new})

        a0 = np.random.choice(range(prob_weights.shape[1]),
                              p=prob_weights.ravel())
        # print(prob_weights)
        a1 = sess.run([self.a_1], {self.s: s})[0]
        a2 = sess.run([self.a_2], {self.s: s})[0]
        # print(a1)
        return a0, a1, a2

    def save_ckpt(self):
        # saver =  tf.train.Saver()
        # saver.save(sess,"model.ckpt")
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params + self.c_params,
                           save_dir=self.scope, printable=False)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params + self.c_params, save_dir=self.scope, printable=False)
        return

    def _build_net(self):


        with tf.variable_scope("actor") as scope:
            self.a_bridge = Util.block(self.s, self.config_a.bridge, "bridge")
            self.mu_1 = Util.block(self.a_bridge, self.config_a.mu_1, "mu_1")
            self.mu_2 = Util.block(self.a_bridge, self.config_a.mu_2, "mu_2")
            self.sigma_1 = Util.block(self.a_bridge, self.config_a.sigma_1, "sigma_1")
            self.sigma_2 = Util.block(self.a_bridge, self.config_a.sigma_2, "sigma_2")
            self.action = Util.block(self.a_bridge, self.config_a.action, "action")
            self.action = tf.multiply(self.action, self.available)
            self.action = self.action + 1e-5  # added to avoid dividing by zero
            self.action = self.action / tf.reduce_sum(self.action, 1, keep_dims=True)

        with tf.variable_scope("critic") as scope:
            self.c_bridge = Util.block(self.s, self.config_c.bridge, "bridge")
            self.value = Util.block(self.c_bridge, self.config_c.value, "value")


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


class Worker:
    def __init__(self, name, globalAC,  config_a, config_c):
        self.name = name
        # self.globalAC = globalAC
        # self.globalAC.load_ckpt()
        self.AC = ACnet(name, globalAC,  config_a, config_c)
        globalAC.load_ckpt()
        self.AC.pull_global()
        self.hard = 0
        self.env = wrap(game[self.hard])

    def pre_process(self, scr, mini, multi, available):
        scr_new = np.zeros_like(scr)
        mini_new = np.zeros_like(mini)
        avail_new = np.zeros([1, available_len_used], dtype=np.float32)
        avail_new[0][0] = 1 if 7 in available else 0
        avail_new[0][1] = 1 if 331 in available else 0
        for i in range(scr_num):
            scr_new[i] = scr[i] - np.mean(scr[i])
            scr_new[i] = scr_new[i] / (np.std(scr_new[i]) + 1e-5)  # preprocessing

            # TODO:this preprocess is not completely the same as Deepmind! HOW TO IMPROVE?

        for i in range(mini_num):
            mini_new[i] = mini[i] - np.mean(mini[i])
            mini_new[i] = mini_new[i] / (np.std(mini_new[i]) + 1e-5)  # preprocessing
        '''
        mini_new = mini - np.ones([7,64,64])*np.mean(mini, axis=(1, 2))
        mini_new = mini_new / (np.std(mini_new, axis=(1, 2)) + 1e-5)
        '''
        multi_new = np.log(multi + 1)  # log to prevent large numbers

        scr_new = scr_new[np.newaxis, :]
        mini_new = mini_new[np.newaxis, :]
        multi_new = multi_new[np.newaxis, :]
        return scr_new, mini_new, multi_new, avail_new

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        # self.AC.pull_global()
        total_step = 1
        buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_r, buffer_avail = [], [], [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state, _, _, info = self.env.reset()  # timestep[0] contains rewards, observations, etc. SEE pysc2 FOR MORE INFO
            ep_r = 0
            while True:
                a0, a1, a2 = self.AC.choose_action([state], [info])
                # print(state)
                action = 1 if a0 == 0 else int(2 + a1 * scr_pixels + a2)
                buffer_s.append([state])
                buffer_avail.append([info])
                buffer_a0.append(a0)
                buffer_a1.append(a1)
                buffer_a2.append(a2)
                state, reward, done, info = self.env.step(action)

                buffer_r.append(reward)
                ep_r += reward
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.value, {self.AC.s: [state]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_  # compute v target
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_v_target, buffer_avail = np.vstack(
                        buffer_s), np.vstack(buffer_a0), np.vstack(buffer_a1), np.vstack(
                        buffer_a2), np.vstack(buffer_v_target), np.vstack(
                        buffer_avail)  # put together into a single array
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a0: buffer_a0,
                        self.AC.a1: buffer_a1,
                        self.AC.a2: buffer_a2,
                        self.AC.v_target: buffer_v_target,
                        self.AC.available: buffer_avail,
                    }

                    test = self.AC.update_global_high(feed_dict)  # update parameters

                    buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_r, buffer_avail = [], [], [], [], [], []
                    self.AC.pull_global()

                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:", GLOBAL_EP,
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                    )
                    GLOBAL_EP += 1
                    # self.globalAC.save_ckpt()
                    # with open("/summary.txt",'w') as f:
                    #    f.write('%.lf' % ep_r)
                    
                    break

    def pre_train(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        # self.AC.pull_global()
        total_step = 1
        buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_r, buffer_avail,buffer_a0_exp,buffer_a1_exp,buffer_a2_exp = [], [], [], [], [], [],[],[],[]
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state, _, _, info = self.env.reset()  # timestep[0] contains rewards, observations, etc. SEE pysc2 FOR MORE INFO
            ep_r = 0
            while True:
                a0,a1,a2 = self.AC.choose_action([state],[info])
                a0_exp, a1_exp, a2_exp = teacher.action(state, info)
                # print(state)
                action = 1 if a0 == 0 else int(2 + a1 * scr_pixels + a2)
                buffer_s.append([state])
                buffer_avail.append([info])
                buffer_a0.append(a0)
                buffer_a1.append(a1)
                buffer_a2.append(a2)
                buffer_a0_exp.append(a0_exp)
                buffer_a1_exp.append(a1_exp)
                buffer_a2_exp.append(a2_exp)
                state, reward, done, info = self.env.step(action)

                buffer_r.append(reward)
                ep_r += reward
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.value, {self.AC.s: [state]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_  # compute v target
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_v_target, buffer_avail,buffer_a0_exp,buffer_a1_exp,buffer_a2_exp = np.vstack(
                        buffer_s), np.vstack(buffer_a0), np.vstack(buffer_a1), np.vstack(
                        buffer_a2), np.vstack(buffer_v_target), np.vstack(
                        buffer_avail),np.vstack(buffer_a0_exp),np.vstack(buffer_a1_exp) ,np.vstack(buffer_a2_exp)   # put together into a single array
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a0: buffer_a0,
                        self.AC.a1: buffer_a1,
                        self.AC.a2: buffer_a2,
                        self.AC.a0_exp: buffer_a0_exp,
                        self.AC.a1_exp: buffer_a1_exp,
                        self.AC.a2_exp: buffer_a2_exp,
                        self.AC.v_target: buffer_v_target,
                        self.AC.available: buffer_avail,
                    }
                    test = self.AC.update_global_high(feed_dict)  # update parameters
                    #closs ,aloss,exp_loss= sess.run([self.AC.c_loss,self.AC.a_loss,self.AC.exp_loss], feed_dict=feed_dict)
                    #print("c_loss:",closs,"a_loss:",aloss,"exp_loss",exp_loss)
                    #sigma_1,sigma_2 = sess.run([self.AC.sigma_1,self.AC.sigma_2],feed_dict = feed_dict)
                    entropy,aloss,td,exp_loss = sess.run([self.AC.entropy,self.AC.a_loss,self.AC.td,self.AC.exp_loss],feed_dict = feed_dict)
                    
                    buffer_s, buffer_a0, buffer_a1, buffer_a2, buffer_r, buffer_avail = [], [], [], [], [], []
                    buffer_a0_exp,buffer_a1_exp,buffer_a2_exp = [],[],[]
                    self.AC.pull_global()

                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:", GLOBAL_EP,
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                    )
                    GLOBAL_EP += 1
                    print("entropy",entropy[0][0],"td",td[0],"exp_loss",exp_loss,"aloss",aloss)
                    # self.globalAC.save_ckpt()
                    # with open("/summary.txt",'w') as f:
                    #    f.write('%.lf' % ep_r)
                    if ep_r>score_high[self.hard] or ep_r <score_low[self.hard]:
                        self.env.close()
                        self.hard = self.hard + 1 if ep_r>score_high[self.hard] else self.hard - 1
                        self.env = wrap(game[self.hard])
                    break


def test():
    from config_a3c import config_a, config_c
    ac = ACnet("Global_Net", None,  config_a, config_c)  # we only need its params
    ac.load_ckpt()
    env = wrap(game[0])
    state, _, done, info = env.reset()
    while True:
        a0, a1, a2 = ac.choose_action([state], [info])
        action = 1 if a0 == 0 else int(2 + a1 * scr_pixels + a2)
        state, reward, done, info = env.step(action)
        if done:
            state, _, done, info = env.reset()


# a=ACnet("Global_Net")

def main(unused_argv):
    global sess
    global OPT_A, OPT_C
    global COORD
    # global GLOBAL_AC
    sess = tf.Session()
    from config_a3c import config_a, config_c
    # test()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

    GLOBAL_AC = ACnet(GLOBAL_NET_SCOPE, None,  config_a, config_c)  # we only need its params

    # tl.layers.initialize_global_variables(sess)
    # sess.run(tf.global_variables_initializer())



    COORD = tf.train.Coordinator()

    tl.layers.initialize_global_variables(sess)
    # GLOBAL_AC.test1.print_params()
    # workers[0].AC.test1.print_params()

    ## start TF threading
    GLOBAL_AC.load_ckpt()

    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'Worker_%i' % i  # worker name
        workers.append(Worker(i_name, GLOBAL_AC,  config_a, config_c))

    worker_threads = []
    for worker in workers:
        job = lambda: worker.pre_train()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    GLOBAL_AC.save_ckpt()
    #plt.plot(GLOBAL_RUNNING_R)
    #plt.show()
    plt.plot(GLOBAL_RUNNING_R)
    plt.savefig("a.jpg")
    reward = np.array(GLOBAL_RUNNING_R,dtype = np.float32)
    reward.tofile("aa.bin")


if __name__ == "__main__":
    app.run(main)








