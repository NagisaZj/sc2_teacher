import multiprocessing, threading, os, shutil
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import pysc2
from pysc2 import agents,env
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2 import lib
from pysc2.env import environment
from absl import flags ,app
import teacher
from sc2_util import wrap
from sc2_util import FLAGS, flags

MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE="Global_Net"
UPDATE_GLOBAL_ITER = 40
scr_pixels=64
scr_num=3
scr_bound=[0,scr_pixels-1]
entropy_gamma=0.005
steps=40
action_speed=8
reward_discount=GAMMA=0.9
LR_A = 2e-4    # learning rate for actor
LR_C = 2e-4    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
N_WORKERS = 1
N_A=2
available_len = 524
available_len_used = 2
save_path = "/models"

class ACnet:
    def __init__(self,scope,globalAC=None,config_a=None, config_c=None):
        self.scope=scope
      
        self.config_a = config_a
        self.config_c = config_c
        if scope == GLOBAL_NET_SCOPE:  #build global net
            with tf.variable_scope(scope):
                self.s=tf.placeholder(tf.float32,[None,scr_pixels,scr_pixels,scr_num],"S")
                self.available=tf.placeholder(tf.float32,[None, available_len_used],"available_actions")
                self._build_net()

                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)

                with tf.name_scope("choose_a"):  #choose actions,do not include a0 as a0 is discrete
                    mu_1, sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                    mu_2, sigma_2 = self.mu_2 * scr_bound[1], self.sigma_2 + 1e-5
                    self.a_1=tf.clip_by_value(tf.squeeze(tf.contrib.distributions.Normal(mu_1,sigma_1).sample(1),axis=0),*scr_bound)
                    self.a_2=tf.clip_by_value(tf.squeeze(tf.contrib.distributions.Normal(mu_2,sigma_2).sample(1),axis=0),*scr_bound)

        else:
            with tf.variable_scope(scope): #else, build local network
                self.s = tf.placeholder(tf.float32, [None, scr_pixels, scr_pixels, scr_num], "S")
                self.available = tf.placeholder(tf.float32, [None, available_len_used], "available_actions")
                self.a0 = tf.placeholder(tf.float32,[None,1],"a0")
                self.a1 = tf.placeholder(tf.float32, [None, 1], "a1")
                self.a2 = tf.placeholder(tf.float32, [None, 1], "a2")
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.a0_in = tf.placeholder(tf.float32, [None, 1], "a0_in")
                self.a1_in = tf.placeholder(tf.float32, [None, 1], "a1_in")
                self.a2_in = tf.placeholder(tf.float32, [None, 1], "a2_in")
                self._build_net()


                td=tf.subtract(self.v_target, self.value, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

            with tf.name_scope('wrap_a_out'):
                self.test = self.sigma_1[0]
                mu_1, sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                mu_2, sigma_2 = self.mu_2 * scr_bound[1], self.sigma_2 + 1e-5

            normal_dist_1 = tf.contrib.distributions.Normal(mu_1, sigma_1)
            normal_dist_2 = tf.contrib.distributions.Normal(mu_2, sigma_2)

            with tf.name_scope("a_loss"):    #build loss function
                loss_0 = tf.reduce_sum(tf.square(tf.subtract(self.a0_in , self.a0)))
                loss_1 = tf.reduce_sum(tf.square(self.a1_in - normal_dist_1.mean())*normal_dist_1.variance())
                loss_2 = tf.reduce_sum(tf.square(self.a2_in - normal_dist_2.mean()) * normal_dist_2.variance())

                self.a_loss = loss_0 + loss_1 + loss_2

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


    def update_global(self, feed_dict):  # run by a local
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s,avail_new):  # run by a local
        prob_weights = sess.run(self.action, feed_dict={self.s:s,
                                                        self.available:avail_new})

        a0 = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        #print(prob_weights)
        a1=sess.run([self.a_1], {self.s:s})[0]
        a2 = sess.run([self.a_2], {self.s:s})[0]
        #print(a1)
        return a0,a1,a2

    def save_ckpt(self):
        #saver =  tf.train.Saver()
        #saver.save(sess,"model.ckpt")
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, printable=False)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params+self.c_params, save_dir=self.scope, printable=False)
        return
    def _build_net(self):

        with tf.variable_scope("actor") as scope:
            self.a_bridge = Util.block(self.s, self.config_a.bridge, "bridge")
            self.mu_1 = Util.block(self.a_bridge,self.config_a.mu_1,"mu_1")
            self.mu_2 = Util.block(self.a_bridge,self.config_a.mu_2,"mu_2")
            self.sigma_1 = Util.block(self.a_bridge,self.config_a.sigma_1,"sigma_1")
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
    def __init__(self,name,globalAC,config_a,config_c):
        self.name=name
        self.AC=ACnet(name,globalAC,config_a,config_c)
        globalAC.load_ckpt()
        self.AC.pull_global()
        self.env= wrap()


    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a0 ,buffer_a1, buffer_a2, buffer_r,buffer_avail,buffer_a0_self = [], [],[],[],[],[],[]
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            state,_,_,info = self.env.reset()  #timestep[0] contains rewards, observations, etc. SEE pysc2 FOR MORE INFO
            ep_r=0
            while True:
                a0,a1,a2 = teacher.action(state,info)
                #print([a0,a1,a2])
                a0_self,_,_ = self.AC.choose_action([state],[info])
                action = 1 if a0 == 0 else int(2 + a1 * scr_pixels + a2)
                buffer_s.append([state])
                buffer_avail.append([info])
                buffer_a0.append(a0)
                buffer_a1.append(a1)
                buffer_a2.append(a2)
                buffer_a0_self.append(a0_self)
                state,reward,done,info = self.env.step(action)
                buffer_r.append(reward)
                ep_r +=  reward
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

                    buffer_s,  buffer_a0, buffer_a1, buffer_a2, buffer_v_target, buffer_avail,buffer_a0_self = np.vstack(
                        buffer_s), np.vstack(buffer_a0), np.vstack(buffer_a1 ), np.vstack(
                        buffer_a2), np.vstack(buffer_v_target), np.vstack(
                        buffer_avail) , np.vstack(
                        buffer_a0_self)  # put together into a single array
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a0: buffer_a0_self,
                        self.AC.a1_in: buffer_a1,
                        self.AC.a2_in: buffer_a2,
                        self.AC.a0_in:buffer_a0,
                        self.AC.v_target: buffer_v_target,
                        self.AC.available: buffer_avail,
                    }

                    test = self.AC.update_global(feed_dict)  # update parameters
                    buffer_s,buffer_a0, buffer_a1, buffer_a2, buffer_r, buffer_avail ,buffer_a0_self= [], [], [], [], [], [],[]
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
                    #self.globalAC.save_ckpt()
                    #with open("/summary.txt",'w') as f:
                    #    f.write('%.lf' % ep_r)
                    break



def test():
    from config_a3c import config_a, config_c
    ac = ACnet("Global_Net",None,config_a,config_c)  # we only need its params
    ac.load_ckpt()
    env = wrap()
    state, _ ,done ,info = env.reset()
    while  True:
        a0, a1, a2 = ac.choose_action([state],[info])
        action = 1 if a0 == 0 else int(2 + a1 * scr_pixels + a2)
        state, reward, done, info = env.step(action)
        if done :
            state, _, done, info = env.reset()




#a=ACnet("Global_Net")

def main(unused_argv):
    global sess
    global OPT_A, OPT_C
    global COORD
    global GLOBAL_AC
    sess = tf.Session()
    from config_a3c import config_a,config_c
    #test()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

    GLOBAL_AC = ACnet(GLOBAL_NET_SCOPE,None,config_a,config_c)  # we only need its params
    tl.layers.initialize_global_variables(sess)
    GLOBAL_AC.load_ckpt()
        #tl.layers.initialize_global_variables(sess)
        #sess.run(tf.global_variables_initializer())

    workers = []
        # Create worker
    for i in range(N_WORKERS):
        i_name = 'Worker_%i' % i   # worker name
        workers.append(Worker(i_name, GLOBAL_AC,config_a,config_c))

    COORD = tf.train.Coordinator()

    
    #GLOBAL_AC.test1.print_params()
    #workers[0].AC.test1.print_params()

    ## start TF threading
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    GLOBAL_AC.save_ckpt()

if __name__=="__main__":

    app.run(main)








