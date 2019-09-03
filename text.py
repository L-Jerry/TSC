#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2019/8/10 13:25 
# @Author : Jerry
# @File : text.py

import traci
import os

laneID = ['1si_0', '1si_1', '1si_2', '2si_0', '2si_1', '2si_2',
          '3si_0', '3si_1', '3si_2', '4si_0', '4si_1', '4si_2']


class Text:
    def __init__(self, process, epoch):
        self.process = process
        self.epoch = epoch
        self.laneID = laneID
        self.travelTime = {}  # 旅行时间（秒）
        self.haltingTime = {}  # 停留时间 （秒）
        self.CO2Emission = {}  # CO2排放量 (mg)
        self.COEmission = {}  # CO排放量（mg）
        self.FuelConsumption = {}  # 能源消耗
        self.getSpeed = {}  # 平均速度
        self.route = {}
        if not os.path.exists('episode'):
            os.mkdir('episode')
        if not os.path.exists('vehicles'):
            os.mkdir('vehicles')

    def open_episode(self):
        self.episode_file = open(f'episode/{self.epoch:03d}_{self.process}.csv', 'w')
        self.episode_file.write('init,color,waitingNumber,waitingTime,passedVehicles,meanSpeed,reward\n')

    @property
    def waiting_number(self):
        halting_number = 0
        for i in self.laneID:
            halting_number += traci.lane.getLastStepHaltingNumber(i)
        return halting_number

    @property
    def waiting_time(self):  # 车道上车辆累计等待时间
        waiting_time1 = 0
        for i in self.laneID:
            waiting_time1 += traci.lane.getWaitingTime(i)
        return waiting_time1

    @property
    def mean_speed(self):  # 车道上车辆的平均速度
        speed = 0
        for i in self.laneID:
            speed += traci.lane.getLastStepMeanSpeed(i)
        speed = speed / len(self.laneID)
        return speed

    def save_episode(self, init, color, passed_vehicles, reward):  # 保存每个episode的信息
        self.episode_file.write(str(init) + ',' +str(color) + ',' +
                                str(self.waiting_number) + ',' + str(self.waiting_time) + ',' +
                                str(passed_vehicles) + ',' + str(self.mean_speed) + ',' + str(reward) + '\n')

    def save_vehicle(self):  # veh->仿真结束时，还在路上的车辆
        vehicle_file = open(f'vehicles/{self.epoch:03d}_{self.process}.csv', 'w')
        vehicle_file.write('ID,travelTime,haltingTime,CO2Emission,COEmission,FuelConsumption,meanSpeed,route,over\n')
        for i in self.travelTime:
            if i in self.veh:
                over = '1'
            else:
                over = '0'
            vehicle_file.write(str(i) + ',' + str(self.travelTime[i]) + ',' + str(self.haltingTime[i]) + ',' +
                               str(self.CO2Emission[i]) + ',' + str(self.COEmission[i]) + ',' +
                               str(self.FuelConsumption[i]) + ',' + str(self.getSpeed[i] / self.travelTime[i]) +
                               ',"' + str(self.route[i]) + '",' + over + '\n')
        vehicle_file.close()

    @property
    def veh(self):  # 当前路网车辆
        vehicles = traci.vehicle.getIDList()
        return vehicles

    def calc_veh_parameters(self):
        for vehicle in self.veh:
            self.haltingTime[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
            if vehicle not in self.travelTime:
                self.route[vehicle] = traci.vehicle.getRoute(vehicle)  # 获取车辆路线
                self.travelTime[vehicle] = 1
                self.CO2Emission[vehicle] = traci.vehicle.getCO2Emission(vehicle)
                self.COEmission[vehicle] = traci.vehicle.getCOEmission(vehicle)
                self.FuelConsumption[vehicle] = traci.vehicle.getFuelConsumption(vehicle)
                self.getSpeed[vehicle] = traci.vehicle.getSpeed(vehicle)
            else:
                self.travelTime[vehicle] += 1
                self.CO2Emission[vehicle] += traci.vehicle.getCO2Emission(vehicle)
                self.COEmission[vehicle] += traci.vehicle.getCOEmission(vehicle)
                self.FuelConsumption[vehicle] += traci.vehicle.getFuelConsumption(vehicle)
                self.getSpeed[vehicle] += traci.vehicle.getSpeed(vehicle)

    def close_text(self):
        self.episode_file.close()
        self.epoch += 1
        self.route = {}
        self.travelTime = {}
        self.haltingTime = {}
        self.CO2Emission = {}
        self.COEmission = {}
        self.FuelConsumption = {}
        self.getSpeed = {}
