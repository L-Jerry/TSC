#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/8/10 14:39 
# @Author : Jerry
# @File : main.py

from simulation import Simulation
from text import Text
from neuralNetwork import NeuralNetwork
import numpy as np
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager
import random


def train_network(ls, nn, lock):
    while True:
        if len(ls) >= 20000:
            lock.acquire()
            indexes = random.sample(range(len(ls)), 64)
            sample_batch = [ls[i] for i in indexes]
            del ls[:len(ls) - 20000]
            lock.release()
            s_j_batch = [b[0] for b in sample_batch]
            s1_j_batch = [b[1] for b in sample_batch]
            last_a_t = [b[2] for b in sample_batch]
            a_t = [b[3] for b in sample_batch]
            r_batch = [b[4] for b in sample_batch]
            s_j1_batch = [b[5] for b in sample_batch]
            s1_j1_batch = [b[6] for b in sample_batch]

            nn.train_network(s_j_batch, s1_j_batch, last_a_t, a_t, r_batch, s_j1_batch, s1_j1_batch)

def start_simulation(ls, nn, process, save, lock):
    INITIAL_EPSILON = 0.01
    FINAL_EPSILON = 0.0007
    epsilon = INITIAL_EPSILON
    ACTIONS = 8
    epoch = 0
    sim = Simulation()
    while True:
        text = Text(process, epoch)
        text.open_episode()
        sim.start_simulation()  # 开始仿真
        init = 0
        current_state = sim.get_state()  # 获取最初车辆状态
        s_t = [current_state for i in range(20)]
        c_state1 = sim.red_signal_time.copy()
        last_at = [0 for i in range(ACTIONS)]  # 最初动作全为零
        c_q_value = nn.get_q_value(s_t, c_state1, last_at)  # 当前状态所有动作的Q值
        sim.current_index = np.argmax(c_q_value)  # 选取Q值最大的动作
        last_at = [0 for i in range(ACTIONS)]
        last_at[sim.current_index] = 1  # 上一个动作
        for i in range(20):
            if sim.yog == 0:
                c_state = s_t.copy()  # 车辆位置信息
                last_at = [0 for i in range(ACTIONS)]
                c_state1 = sim.red_signal_time.copy()  # 信号灯红灯持续时间
                c_q_value = nn.get_q_value(c_state, c_state1, last_at)  # 当前状态所有动作的Q值
                sim.next_index = np.argmax(c_q_value)
                sim.yog += 1
                if sim.next_index != sim.current_index:
                    sim.yog = 4
            if sim.yog > 1:  # 执行黄灯相位
                sim.change_to_yellow()
                sim.yog -= 1
            elif sim.yog == 1:  # 执行绿灯相位
                sim.change_to_green()
                sim.yog -= 1
            sim.simulation_step()  # 仿真时间经过1秒
            sim.get_last_vehicle_ID()  # 获取当前时间车辆id
            current_state = sim.get_state()  # 获取当前时间车辆位置状态
            init += 1
            s_t = [*s_t[1:], current_state]
            text.calc_veh_parameters()
            if sim.yog == 0:
                sim.current_index = sim.next_index

        while init < 6020:
            if sim.yog == 0:
                c_state = s_t.copy()  # 车辆位置信息
                last_at = [0 for i in range(ACTIONS)]
                c_state1 = sim.red_signal_time.copy()  # 信号灯红灯持续时间
                c_q_value = nn.get_q_value(c_state, c_state1, last_at)  # 当前状态所有动作的Q值
                sim.next_index = np.argmax(c_q_value)
                if np.random.random() <= epsilon:  # ε-greedy
                    sim.next_index = np.random.randint(ACTIONS)
                sim.yog += 1
                if sim.next_index != sim.current_index:
                    sim.yog = 4

            if sim.yog > 1:  # 执行黄灯相位
                sim.change_to_yellow()
                sim.yog -= 1
            elif sim.yog == 1:  # 执行绿灯相位
                sim.change_to_green()
                sim.yog -= 1
            sim.simulation_step()  # 仿真时间经过1秒
            text.save_episode(init, sim.color, sim.throughputs[-1], sum(sim.throughputs[-20:]))
            sim.get_last_vehicle_ID()  # 获取当前时间车辆id
            current_state = sim.get_state()  # 获取当前时间车辆位置状态
            init += 1
            s_t = [*s_t[1:], current_state]
            text.calc_veh_parameters()
            if sim.yog == 0:
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 99999
                next_state = s_t.copy()  # 状态1
                next_state1 = sim.red_signal_time.copy()  # 状态2
                v_a_t0 = [0 for i in range(ACTIONS)]
                a_t0 = sim.current_index  # 状态3
                v_a_t0[a_t0] = 1
                v_a_t1 = [0 for i in range(ACTIONS)]
                a_t = sim.next_index  # 动作、状态2
                v_a_t1[a_t] = 1
                r_t = sum(sim.throughputs[-20:]) - np.sqrt(sum(next_state1))
                # 前20秒通过的车辆-sqrt(sum(红灯时长-300, 0))  ↑
                lock.acquire()
                ls.append((c_state, c_state1, v_a_t0, v_a_t1, r_t, next_state, next_state1))
                lock.release()
                sim.current_index = sim.next_index

        text.save_vehicle()
        text.close_text()
        epoch += 1
        lock.acquire()
        with open('throughput.csv', 'a+') as f:
            f.write(f'{epoch},{process},{sum(sim.throughputs)},{sum(sim.all_throughputs)}\n')
        lock.release()
        sim.stop_simulation()
        if (epoch - 1) % 10 == 0 and save:
            nn.save_network(epoch)


if __name__ == '__main__':
    with open('throughput.csv', 'w') as f:
        f.write('epoch,process,throughput,allThroughtput\n')
    BaseManager.register('NeuralNetwork', NeuralNetwork)
    manager = BaseManager()
    manager.start()
    nn = manager.NeuralNetwork()
    ls = Manager().list()  # 保存所有进程的数据
    lock = Lock()  # 进程锁
    processes = [Process(target=train_network, args=(ls, nn, lock))]
    # processes.append(Process(target=start_simulation, args=(ls, nn, 0, True, lock)))
    for i in range(12):
        save = False
        if i == 0:
            save = True
        p = Process(target=start_simulation, args=(ls, nn, i, save, lock))
        processes.append(p)
    for process in processes:
        process.start()
    for process in processes:
        process.join()

