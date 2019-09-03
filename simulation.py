#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/8/7 10:23
# @Author : Jerry
# @File : simulation.py

import traci
import random
import numpy as np
import os


dict1 = {0: 'GGGGrrGrrGrr',
         1: 'GrrGGGGrrGrr',
         2: 'GrrGrrGGGGrr',
         3: 'GrrGrrGrrGGG',
         4: 'GrGGrrGrGGrr',
         5: 'GGrGrrGGrGrr',
         6: 'GrrGrGGrrGrG',
         7: 'GrrGGrGrrGGr'}  # All the situation of phases 8种相位

yellow = {}  # 黄灯方案
for i in dict1:
    for j in dict1:
        if i == j:
            continue
        colortmp = ''
        for k in range(len(dict1[i])):
            if dict1[i][k] == 'r':
                colortmp += 'r'
            elif dict1[i][k] == 'G' and dict1[j][k] == 'r':
                colortmp += 'y'
            else:
                colortmp += 'G'
        yellow['%02d%02d' % (i, j)] = colortmp

signals = {0: 1, 1: 2, 2: 0, 8: 1, 9: 2, 10: 0}  # 0,8：车辆右转信号，1,9：车辆直行信号，2,10：车辆左转信号

class Simulation:
    def __init__(self):
        self.epoch = 0
        self.vehID = 0
        self.sumocfg = ['sumo', "-c", 'cfg/net.sumo.cfg', "--waiting-time-memory", "6000", "--start"]  #
        self.router = ['3si 2o', '3si 1o', '3si 4o', '1si 2o', '1si 3o', '1si 4o',
                       '2si 1o', '2si 3o', '2si 4o', '4si 1o', '4si 2o', '3si 3o']
        self.entry = ['1si_1', '1si_2', '2si_1', '2si_2', '3si_1', '3si_2', '4si_1', '4si_2']
        self.red_signal = [0 for i in range(12)]  # 红灯时长
        self.color = dict1[0]
        self.current_index = 0
        self.next_index = None
        self.yog = 0
        self.throughputs = []  # 不包括右转通过的车辆
        self.all_throughputs = []  # 包括右转通过的车辆
        self.current_vehicle_ID = set()
        self.last_vehicle_ID = set()
        self.all_current_vehicle_ID = set()
        self.all_last_vehicle_ID = set()

    def start_simulation(self):  # 创建不同的路径
        traci.start(self.sumocfg)
        for i, j in enumerate(self.router):
            rou = j.split(' ')
            traci.route.add(str(i), rou)

    def get_last_vehicle_ID(self):
        self.last_vehicle_ID = self.current_vehicle_ID
        self.all_last_vehicle_ID = self.all_current_vehicle_ID

    def calc_throughputs(self):
        # 前一时间的直行和左转车辆-下一时间直行、左转、右转车辆
        throughputs = len(self.last_vehicle_ID - self.all_current_vehicle_ID)
        all_throughts = len(self.all_last_vehicle_ID - self.all_current_vehicle_ID)
        self.throughputs.append(throughputs)
        self.all_throughputs.append(all_throughts)

    @property
    def get_vehicle_ID(self):  # 获取入口车辆ID
        a = set(traci.lane.getLastStepVehicleIDs('1si_0'))
        b = set(traci.lane.getLastStepVehicleIDs('1si_1'))
        c = set(traci.lane.getLastStepVehicleIDs('1si_2'))
        d = set(traci.lane.getLastStepVehicleIDs('2si_0'))
        e = set(traci.lane.getLastStepVehicleIDs('2si_1'))
        f = set(traci.lane.getLastStepVehicleIDs('2si_2'))
        g = set(traci.lane.getLastStepVehicleIDs('3si_0'))
        h = set(traci.lane.getLastStepVehicleIDs('3si_1'))
        i = set(traci.lane.getLastStepVehicleIDs('3si_2'))
        j = set(traci.lane.getLastStepVehicleIDs('4si_0'))
        k = set(traci.lane.getLastStepVehicleIDs('4si_1'))
        l = set(traci.lane.getLastStepVehicleIDs('4si_2'))
        self.current_vehicle_ID = b | c | e | f | h | i | k | l
        self.all_current_vehicle_ID = a | b | c | d | e | f | g | h | i | j | k | l
        return a, b, c, d, e, f, g, h, i, j, k, l

    def get_state(self):  # 获取当前车辆位置状态
        mat = np.array([[[0, 0, 0] for i in range(50)] for j in range(12)])
        for i, j in enumerate(self.get_vehicle_ID):
            for vehID in j:
                pos = traci.vehicle.getPosition(vehID)
                signal = traci.vehicle.getSignals(vehID)
                signalpos = signals[signal]
                if i == 0 or i == 1 or i == 2:
                    y = int(pos[0] / 2)
                    mat[i, y, signalpos] = 1
                elif i == 3 or i == 4 or i == 5:
                    y = int((200 - pos[0]) / 2)
                    mat[i, y, signalpos] = 1
                elif i == 6 or i == 7 or i == 8:
                    y = int(pos[1] / 2)
                    mat[i, y, signalpos] = 1
                else:
                    y = int((200 - pos[1]) / 2)
                    mat[i, y, signalpos] = 1
        return mat

    def get_red_time(self):  # 计算当前信号灯的红灯持续时间
        for i in range(len(self.color)):
            if self.color[i] == 'r':
                self.red_signal[i] += 1
            else:
                self.red_signal[i] = 0

    @property  # max(红灯持续时间减去最大忍受时间，0)
    def red_signal_time(self):
        time = [0 for i in range(12)]
        for i in range(len(self.red_signal)):
            time[i] = max(self.red_signal[i] - 300, 0)
        return time

    def simulation_step(self):  # 选择信号灯颜色并执行1秒
        num = random.randint(1, 12)
        for i in range(num):
            rand = random.randint(0, len(self.router) - 1)
            traci.vehicle.add(str(self.vehID), str(rand))
            self.vehID += 1
        traci.trafficlight.setRedYellowGreenState('0', self.color)
        traci.simulationStep()
        self.calc_throughputs()
        self.get_red_time()

    def change_to_yellow(self):
        self.color = yellow['%02d%02d' % (self.current_index, self.next_index)]

    def change_to_green(self):
        self.color = dict1[self.next_index]

    def stop_simulation(self):
        traci.close(False)
        self.vehID = 0
        self.epoch += 1
        self.throughputs = []
        self.all_throughputs = []
        self.red_signal = [0 for i in range(12)]  # 红灯时长
        self.current_vehicle_ID = set()
        self.last_vehicle_ID = set()
        self.all_current_vehicle_ID = set()
        self.all_last_vehicle_ID = set()
        self.yog = 0


if __name__ == '__main__':
    sim = Simulation()
    sim.start_simulation()
    for i in range(8):
        for j in range(200):
            sim.simulation_step()
        if i != 7:
            color = yellow['%02d%02d' % (i, i+1)]
            for j in range(100):
                sim.simulation_step()
    sim.stop_simulation()
