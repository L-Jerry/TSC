交通信号灯单路口控制<br>
状态1：驶入交叉口车道的车辆位置以及车辆左转、右转、直行信号<br>
状态2：sum(12个信号灯红灯持续时间 - 红灯最大忍受时间, 0)<br>
状态3：上一个信号灯的相位<br>
动作：8种不同相位，每秒中进行一次判断<br>
奖励：前20秒通过路口的车辆数减去sqrt(sum(max(12个信号灯红灯持续时间减去红灯最大忍受时间, 0)))<br>
神经网络输入： 前20秒的状态1，当前状态2，当前状态3  lstm多对一<br>
