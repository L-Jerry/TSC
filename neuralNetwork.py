#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/8/7 10:23
# @Author : Jerry 
# @File : neuralNetwork.py

import tensorflow as tf
import numpy as np
import random
import os

ACTIONS = 8
GAMMA = 0.99


class NeuralNetwork:
    def __init__(self):
        self.h_size = 2560  # lstm网络cell的个数
        self.sess = tf.InteractiveSession()
        self.s_t = tf.placeholder("float", [None, 20, 12, 50, 3])  # 车辆位置状态
        self.s1_t = tf.placeholder("float", [None, 12])  # 红灯时长
        self.last_a = tf.placeholder('float', [None, ACTIONS])  # 上一个信号灯的动作
        self.readout = self.build_net()  # 网络结构
        self.y = tf.placeholder('float', [None])  # 标签
        self.a = tf.placeholder('float', [None, ACTIONS])  # 动作
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)  # 当前动作的Q值
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))  # TD Error
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if not os.path.exists('saved_networks'):
            os.mkdir('saved_networks')

    def build_net(self):  # 构建神经网络
        s_t_flatten = tf.reshape(self.s_t, [-1, 20, 1800])  # 展平
        lstmCell = tf.keras.layers.LSTMCell(self.h_size)
        value, _ = tf.nn.dynamic_rnn(lstmCell, s_t_flatten, dtype='float')
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        w1_1 = self.weight_variable([self.h_size, 10240])
        b1_1 = self.bias_variable([10240])
        w1_2 = self.weight_variable([12, 10240])
        w1_3 = self.weight_variable([8, 10240])
        w2 = self.weight_variable([10240, 2560])
        b2 = self.bias_variable([2560])
        w3 = self.weight_variable([2560, ACTIONS])
        b3 = self.bias_variable([ACTIONS])
        fc_1 = tf.nn.relu(tf.matmul(last, w1_1) + tf.matmul(self.s1_t, w1_2) + tf.matmul(self.last_a, w1_3) + b1_1)
        fc_2 = tf.nn.relu(tf.matmul(fc_1, w2) + b2)
        readout = tf.matmul(fc_2, w3) + b3
        return readout

    def weight_variable(self, shape):  # 神经网络权重
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):  # 神经网络偏置
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def ckpt(self):  # 加载保存的神经网络
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights!")

    def train_network(self, s_j_batch,  # 状态1
                      s1_j_batch,  # 状态2
                      last_a_t,  # 状态3
                      a_t,  # 下一步状态3以及动作
                      r_batch,  # 奖励
                      s_j1_batch,  # 下一步状态1
                      s1_j1_batch):  # 下一步状态2

        y_batch = []
        readout_j1_batch = self.sess.run(self.readout, feed_dict={self.s_t: s_j1_batch,
                                                                  self.s1_t: s1_j1_batch,
                                                                  self.last_a: a_t})
        for i in range(len(s_j_batch)):
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

        self.sess.run(self.train_step, feed_dict={
            self.y: y_batch,
            self.a: a_t,
            self.s_t: s_j_batch,
            self.s1_t: s1_j_batch,
            self.last_a: last_a_t
        })

    def get_q_value(self, c_state, c_state1, last_at):  # 返回当前状态下的Q值
        c_q_value = self.sess.run(self.readout, feed_dict={self.s_t: [c_state],
                                                           self.s1_t: [c_state1],
                                                           self.last_a: [last_at]})[0]
        return c_q_value

    def save_network(self, t):  # 保存神经网络
        self.saver.save(self.sess, 'saved_networks/sumo-drqn', global_step=t)


if __name__ == '__main__':
    nn = NeuralNetwork()
    # minibatch = []
    # s_0 = [random.random() for i in range(20 * 12 * 50 * 3)]
    # s_0 = [np.array(s_0).reshape((20, 12, 50, 3))]
    # s1_0 = [np.array([1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    # a = [3]
    # r = [4]
    # s_1 = [random.random() for i in range(20 * 12 * 50 * 3)]
    # s_1 = [np.array(s_1).reshape((20, 12, 50, 3))]
    # s1_1 = [np.array([1, 2, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0])]
    # nn.train_network(s_0, s1_0, a, a, r, s_1, s1_1)
