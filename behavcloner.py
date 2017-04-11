#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:07:23 2017

@author: etienneperot
"""

import pickle, tensorflow as tf, tf_util, numpy as np
import load_policy
import gym
import time

TENSORBOARD=False
LOAD_CHECKPOINT=False
SAVE_MODELS=False
LEARNING_RATE_START=5e-4
LEARNING_RATE_END=5e-5
BATCH_SIZE = 64
NORMX=False
MAX_SIZE = 5000

import sklearn
from sklearn import preprocessing

def shuffle_cut(x,y):
    x, y = sklearn.utils.shuffle(x, y, random_state=0)
    x_ = x[:MAX_SIZE]
    y_ = y[:MAX_SIZE]
    return x_, y_

def prepare_data(x,y, N_train, ppsx=None, ppsy=None):
    x_train = x[:N_train]
    y_train = y[:N_train]
    
    x_test = x[N_train:-1]
    y_test = y[N_train:-1]
    

    x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)
    
    if ppsx != None:
        xtrn = ppsx.fit_transform(x_train)
        xten = ppsx.transform(x_test)
    else:
        xtrn = x_train
        xten = x_test
        
    if ppsy != None:      
        ytrn = ppsy.fit_transform(y_train)
        yten = ppsy.transform(y_test)
    else:
        ytrn = y_train
        yten = y_test
    
    return xtrn, ytrn, xten, yten

class BehavCloner:
    def __init__(self, device, num_descriptors, num_actions):
        self.D = num_descriptors
        self.A = num_actions
        self.device = device
        self.learning_rate = LEARNING_RATE_START
        self.graph = tf.Graph()
        self.ppsx = None
        self.ppsy = None
        if NORMX:
            self.ppsx = preprocessing.StandardScaler()

    
        with self.graph.as_default():
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                if TENSORBOARD: self._create_tensor_board()
                if LOAD_CHECKPOINT or SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                    
                    
    def _dense(self,input, out_dim, name, func=None):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
            
            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output
 
    def _dense_bn(self, input, size, name, is_training, func):
        x1 = self._dense(input, 256, name=name)
        x2 = self.batch_norm_layer(x1,is_training,name+'_bn',func)
        return x2
   
    def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
        return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
		updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
		updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

    def _create_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.D], name='obervations')
        self.y = tf.placeholder(tf.float32, [None, self.A], name='actions')
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.is_training = tf.placeholder(tf.bool)
        
        #very simple model
        self.d1 = self._dense_bn(self.x, 256, 'dense1', self.is_training, tf_util.lrelu)
        self.d2 = self._dense_bn(self.d1, 256, 'dense2', self.is_training, tf_util.lrelu) 
        self.logits_p = self._dense(self.d2, self.A, 'logits_p', None)
        
        #loss
        self.loss = tf.reduce_sum(tf.square(self.logits_p - self.y) ) / BATCH_SIZE
        self.opt = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)
     
    def _eval(self, x, y):
         loss = self.sess.run([self.loss], feed_dict={self.x: x, self.y: y, self.is_training: False})
         return loss
     
    def _predict_p(self, x):
         p = self.sess.run([self.logits_p], feed_dict={self.x: x, self.is_training: False})
         return p
    
    def _train(self, x, y):
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x: x, self.y: y, self.is_training: True})
        train_op, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
    
    def __get_base_feed_dict(self):
        return {self.var_learning_rate: self.learning_rate, self.is_training: True}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

def test_bc(x,y):
    N_test = x.shape[0]
    eval_loss = 0
    n_test_iter = int(N_test / BATCH_SIZE)
    for i in range(n_test_iter):
        j = (i*BATCH_SIZE)%(N_test-BATCH_SIZE)
        x = x[j:j+BATCH_SIZE]
        y = y[j:j+BATCH_SIZE]
        eval_loss += bc1._eval(x,y)[0] / n_test_iter
    return eval_loss

def train_bc(agent, x, y, niter=10000, eval_freq = 1000):
    print('Start Training Agent')
    EPISODE = 10
    ANNEALING_EPISODE_COUNT = niter
    learning_rate_multiplier = (LEARNING_RATE_END - LEARNING_RATE_START) / ANNEALING_EPISODE_COUNT
                            
    Ntrain = int(x.shape[0] * 0.7)                       
    xtrn, ytrn, xten, yten = prepare_data(x, y, Ntrain, agent.ppsx)
    
    for i in range(niter):
        j = (i*BATCH_SIZE)%(Ntrain-BATCH_SIZE)
        x_ = xtrn[j:j+BATCH_SIZE]
        y_ = ytrn[j:j+BATCH_SIZE]
    
        train_loss = bc1._train(x_,y_)
 
        if i%EPISODE == 0:
            step = min(i, ANNEALING_EPISODE_COUNT - 1)
            bc1.learning_rate = LEARNING_RATE_START + learning_rate_multiplier * step
            #print('Train Loss = ',train_loss,'; lr = ',bc1.learning_rate )
          
        if i%eval_freq == 0:
            eval_loss = test_bc(xten,yten)
            print('iter=',i,'; Test Loss = ', eval_loss)


    
#Eval Policy during one episode 
def run_policy(agent, envname, num_rollouts=10,ppsx=None, ppsy=None):
    envname = 'Humanoid-v1'
    env = gym.make(envname)
    steps = 0
    for i in range(num_rollouts):
        print('Test#',i)
        obs = env.reset()
        done = False
        totalr = 0.
        while not done:
            if ppsx != None:
                x = ppsx.transform(obs[None,:])
            else:
                x = obs[None,:]
            action = agent._predict_p(x)
            if ppsy != None:
                action = ppsy.inverse_transform(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            env.render()
            time.sleep(0.005)
            steps += 1
        print('Total Reward during Run = ',totalr)
        
    
def dagger_collect(agent, expert, envname, old_data, num_rollouts=2):
    print('Start Dagger Collect...')
    envname = 'Humanoid-v1'
    env = gym.make(envname)
    max_steps = 10000 #env.spec.timestep_limit

    totalr = 0.
    observations = []
    #actions = []
    expert_actions = []
    with tf.Session():
        #tf_util.initialize()
        for i in range(num_rollouts):
            steps = 0
            obs = env.reset()
            done = False
            while not done:
                if agent.ppsx != None:
                    x = agent.ppsx.transform(obs[None,:])
                else:
                    x = obs[None,:]
                action = agent._predict_p(x)
                
                expert_action =  expert(obs[None,:])
                
                #...collect
                obs, r, done, _ = env.step(expert_action) #should be action
                totalr += r
                
                #env.render()
                #time.sleep(0.005)
            
            
                observations.append(obs)
                #actions.append(action)
                expert_actions.append(expert_action)
                
                steps += 1
                if steps >= max_steps:
                    break
        
    x = old_data['observations']
    y = old_data['actions']
    
    x = np.concatenate((x,np.array(observations)))
    y = np.concatenate((y,np.array(expert_actions)))
    
    #limit max_size, would be good to keep in function of some metric with expert actions...
    if MAX_SIZE != -1:
        x, y = shuffle_cut(x,y)
    
    new_data = {}
    new_data['observations'] = x
    new_data['actions'] = y   
    return new_data

#data
expert_data = pickle.load( open( "expert_data.p", "rb")  )
x = expert_data['observations']
y = expert_data['actions'].squeeze()
x, y = shuffle_cut(x,y)


N = x.shape[0]
D = x.shape[1]
A = y.shape[-1]

print('N,D,A',N,D,A)

bc1 = BehavCloner(device='cpu:0',num_descriptors=D,num_actions=A)

#niter = 400
#train_bc(bc1, x, y, niter)
#run_policy(bc1, 'Humanoid-v0', ppsx=bc1.ppsx, num_rollouts=2)
'''
    Dagger 
    (feed-forward)
            0. Init Data A on Expert Policy
            1. Train Agent on Data A 
            2. Collect Data from Agent Data B (agent policy, expert actions)
            3. Aggregate A=A+B & Go back to 1.
'''

expert_policy_file = './experts/Humanoid-v1.pkl'
expert = load_policy.load_policy(expert_policy_file)
niter = 10
data = expert_data
for i in range(20):
    #1. 
    train_bc(bc1, x, y, MAX_SIZE, 1000)
    #2. 
    data = dagger_collect(bc1, expert, envname='Humanoid-v1', old_data=data, num_rollouts=100)
    #3. Test Agent new policy
    run_policy(bc1, 'Humanoid-v1', ppsx=bc1.ppsx, num_rollouts=2)

