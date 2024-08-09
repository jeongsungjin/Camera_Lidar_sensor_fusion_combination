#!/usr/bin/env python

import rospy
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32, Int32
import socket
import math
import time
import threading

class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error


class MotorController:
    def __init__(self):
        rospy.init_node('motor_controller', anonymous=True)
        
        # LiDAR 및 카메라 데이터에 대한 Subscriber
        self.lidar_sub = rospy.Subscriber('/detected_objects', MarkerArray, self.lidar_callback)
        self.camera_sub = rospy.Subscriber('/lane_x_location', Float32, self.camera_callback)
        
        # 제어 명령을 위한 Publisher
        self.motor_pub = rospy.Publisher('/motor_cmd', Int32, queue_size=1)
        self.servo_pub = rospy.Publisher('/servo_cmd', Int32, queue_size=1)

        # 초기 상태 변수 설정
        self.x_location = None
        self.obstacle_list = []
        self.motor_pwm = 1580  # 기본 모터 PWM
        self.servo_pwm = 1500  # 기본 서보 PWM

        rospy.loginfo("Motor Controller Node Initialized")

    def lidar_callback(self, msg):
        # LiDAR 데이터 처리 (발견된 장애물)
        self.obstacle_list = []
        for marker in msg.markers:
            obstacle = Obstacle(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list.append(obstacle)
        
        # 장애물 거리로 정렬
        self.obstacle_list.sort(key=lambda obstacle: obstacle.distance())
        self.process_data()
    
    def camera_callback(self, data):
        # 카메라의 x 위치 데이터 처리
        self.x_location = data.data
        self.process_data()
    
    def process_data(self):
        if self.x_location is not None and self.obstacle_list:
            # 최신 데이터 가져오기
            x_location = self.x_location
            closest_obstacle = self.obstacle_list[0]
            pid = PID(1.0, 0.05, 0.1)
            # PID 제어 예제 처리
            motor_speed = 5
            self.angle = pid.pid_control(x_location - 256)
            max_angle = 40.0
            self.angle = max(min(self.angle, max_angle), -max_angle)

            self.servo_pwm = self.angle_to_pwm(self.angle)
            self.motor_pwm = 1580

            if closest_obstacle.distance() < 1:
                motor_speed = 0
                self.motor_pwm = 1400

            rospy.loginfo("Steering Angle: {} degrees".format(self.angle))
            rospy.loginfo("Motor Speed: {}".format(motor_speed))
            rospy.loginfo("가장 가까운 장애물 거리: {}".format(closest_obstacle.distance()))
            rospy.loginfo("조향 pwm : {}".format(self.servo_pwm))
            rospy.loginfo("모터 pwm : {}".format(self.motor_pwm))

            # 제어 명령 전송
            self.publish_control_cmd(self.motor_pwm, self.servo_pwm)

    def angle_to_pwm(self, angle):
        # 각도 (-45도에서 45도)를 서보 PWM 범위 (1100에서 1900)으로 매핑
        min_angle = -45
        max_angle = 45
        min_pwm = 1100
        max_pwm = 1900

        # 각도를 [0, 1] 범위로 정규화
        normalized_angle = (angle - min_angle) / (max_angle - min_angle)
        # 정규화된 각도를 PWM 범위로 매핑
        pwm_value = min_pwm + (max_pwm - min_pwm) * normalized_angle
        return int(pwm_value)
    
    def send_messages(self, client_socket, server_host, server_port):
        try:
            client_socket.connect((server_host, server_port))
            rospy.loginfo("Connected to {}:{}".format(server_host, server_port))

            while True:
                # motor_pwm과 servo_pwm을 메시지로 생성
                message = '{},{}'.format(self.motor_pwm, self.servo_pwm)
                client_socket.sendall(message.encode())
                rospy.loginfo("Sent to {}:{}: {}".format(server_host, server_port, message))
                time.sleep(0.1)  # 0.1초 대기
        except (socket.error, socket.timeout) as e:
            rospy.logwarn("Connection lost with {}:{}, retrying... Error: {}".format(server_host, server_port, e))
            time.sleep(3)  # 재연결 전 대기
        finally:
            client_socket.close()
    
    def publish_control_cmd(self, motor_cmd, servo_cmd):
        # ROS 토픽에 모터 및 서보 명령 발행
        self.motor_pub.publish(int(motor_cmd))
        self.servo_pub.publish(int(servo_cmd))

    def run(self):
        rospy.spin()

    def client_thread(self, server_host, server_port, client_ip):
        self.client_ip = client_ip
        while True:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.send_messages(client_socket, server_host, server_port)


if __name__ == '__main__':
    SERVER_HOST_1 = '192.168.2.158'  # First server's IP address
    SERVER_PORT_1 = 1234            # Port for the first server

    CLIENT_IP_1 = '192.168.2.168'    # First client interface IP address

    # Create and start a thread for communication with the first server
    controller = MotorController()
    
    thread_1 = threading.Thread(target=controller.client_thread, args=(SERVER_HOST_1, SERVER_PORT_1, CLIENT_IP_1))

    thread_1.start()
    
    
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
