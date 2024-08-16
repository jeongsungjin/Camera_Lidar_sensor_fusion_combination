#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import pigpio
import time
import sys
import select

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

        # pigpio 초기화
        self.pi = pigpio.pi()
        if not self.pi.connected:
            rospy.logerr("pigpio 데몬에 연결할 수 없습니다. 데몬을 실행하고 다시 시도해 주세요.")
            rospy.signal_shutdown("pigpio 데몬에 연결할 수 없음")

        # 카메라 데이터에 대한 Subscriber
        self.camera_sub = rospy.Subscriber('/lane_x_location', Float32, self.camera_callback)

        #gps 좌표 받아오기


        # 초기 상태 변수 설정
        self.x_location = 256
        self.motor_pwm = 1500  # 기본 모터 PWM
        self.servo_pwm = 1480  # 기본 서보 PWM

        # GPIO 핀 번호 (BCM 모드)
        self.STEERING_SERVO_PIN = 17  # 서보모터 신호 핀 (조향용)
        self.ESC_PIN = 18             # ESC 신호 핀 (속도 제어용)

        # 서보모터의 안전한 기본 펄스 폭 범위 (마이크로초)
        self.STEERING_MIN_PULSE_WIDTH = 1200
        self.STEERING_MAX_PULSE_WIDTH = 1800
        self.STEERING_CENTER_PULSE_WIDTH = (self.STEERING_MIN_PULSE_WIDTH + self.STEERING_MAX_PULSE_WIDTH) / 2

        # ESC의 안전한 기본 펄스 폭 범위 (마이크로초)
        self.ESC_MIN_PULSE_WIDTH = 1300
        self.ESC_MAX_PULSE_WIDTH = 1700
        self.ESC_STOP_PULSE_WIDTH = 1500

        # 시나리오 플래그 변수 초기화
        self.avoidance_flag = False
        self.deceleration_flag = False
        self.stop_flag = True

        # 회피 시나리오 제어 플래그
        self.avoidance_phase = 0
        self.avoidance_start_time = None

        rospy.loginfo("Motor Controller Node Initialized")

    def camera_callback(self, data):
        self.x_location = data.data

    def process_data(self):
        if self.x_location is not None:
            pid = PID(1.0, 0.003, 0.03 )  # PID 제어기 초기화

            # PID 제어를 통해 각도 계산
            angle = pid.pid_control(self.x_location - 320)
            max_angle = 30.0
            angle = max(min(angle, max_angle), -max_angle)

            # 각도를 PWM 값으로 변환
            self.servo_pwm = self.angle_to_pwm(angle)
            # self.servo_pwm = 1480 #조향 고정 @@@@@@@
            # 시나리오 플래그에 따른 모터 제어
            if self.avoidance_flag:
                rospy.loginfo("Avoidance Scenario Activated")
                self.handle_avoidance_scenario()

            elif self.deceleration_flag:
                rospy.loginfo("Deceleration Scenario Activated")
                self.motor_pwm = 1570  # 감속 

            elif self.stop_flag:
                rospy.loginfo("Stop and Go Scenario Activated")
                self.motor_pwm = 1500  # 정지 
                
                #비깜 켜기

            else:
                self.motor_pwm = 1600  # 기본 주행

            rospy.loginfo("Steering Angle: {} degrees".format(angle))
            rospy.loginfo("Motor Speed: {}".format(self.motor_pwm))
            rospy.loginfo("조향 pwm : {}".format(self.servo_pwm))
            rospy.loginfo("모터 pwm : {}".format(self.motor_pwm))
            
            # 서보모터와 ESC 설정
            self.set_steering_servo_pulsewidth(self.servo_pwm)
            time.sleep(0.1)  # 짧은 지연 추가
            self.set_esc_pulsewidth(self.motor_pwm)

    def handle_avoidance_scenario(self):
        # 회피 시나리오
        rate = rospy.Rate(15)  # 주사율 15Hz
        loop_duration = 20  #루프 카운트

        # 좌측 조향 1초 유지
        self.servo_pwm = self.angle_to_pwm(-25)  # 좌측 조향 
        self.motor_pwm = 1580  
        for _ in range(loop_duration):
            self.set_steering_servo_pulsewidth(self.servo_pwm)
            self.set_esc_pulsewidth(self.motor_pwm)
            rate.sleep()

        # 우측 조향 1초 유지
        self.servo_pwm = self.angle_to_pwm(24)  # 우측 조향 
        for _ in range(loop_duration):
            self.set_steering_servo_pulsewidth(self.servo_pwm)
            self.set_esc_pulsewidth(self.motor_pwm)
            rate.sleep()

        # 회피 시나리오 종료
        self.avoidance_flag = False
         
    def angle_to_pwm(self, angle):
        min_angle = -30
        max_angle = 30
        min_pwm = 1200
        max_pwm = 1800

        normalized_angle = (angle - min_angle) / (max_angle - min_angle)
        pwm_value = min_pwm + (max_pwm - min_pwm) * normalized_angle
        return int(pwm_value)

    def set_steering_servo_pulsewidth(self, pulsewidth):
        pulsewidth = max(self.STEERING_MIN_PULSE_WIDTH, min(self.STEERING_MAX_PULSE_WIDTH, pulsewidth))
        # rospy.loginfo(f"Setting steering servo pulsewidth: {pulsewidth}")
        self.pi.set_servo_pulsewidth(self.STEERING_SERVO_PIN, pulsewidth)

    def set_esc_pulsewidth(self, pulsewidth):
        pulsewidth = max(self.ESC_MIN_PULSE_WIDTH, min(self.ESC_MAX_PULSE_WIDTH, pulsewidth))
        # rospy.loginfo(f"Setting ESC pulsewidth: {pulsewidth}")
        self.pi.set_servo_pulsewidth(self.ESC_PIN, pulsewidth)
    
    def run(self):
        rate = rospy.Rate(15)  # 주사율 15Hz
        while not rospy.is_shutdown():
            self.check_scenario_flags()  # 시나리오 플래그 체크
            self.process_data()
            rate.sleep()

    def check_scenario_flags(self):
        # 비차단 키보드 입력 읽기
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == '1':
                self.avoidance_flag = True
                self.deceleration_flag = False
                self.stop_flag = False
                self.avoidance_phase = 0
            elif key == '2':
                self.avoidance_flag = False
                self.deceleration_flag = True
                self.stop_flag = False
            elif key == '3':
                self.avoidance_flag = False
                self.deceleration_flag = False
                self.stop_flag = True
            elif key == '0':
                self.avoidance_flag = False
                self.deceleration_flag = False
                self.stop_flag = False

    def cleanup(self):
        rospy.loginfo("Cleaning up...")
        self.pi.set_servo_pulsewidth(self.STEERING_SERVO_PIN, self.STEERING_CENTER_PULSE_WIDTH)
        self.pi.set_servo_pulsewidth(self.ESC_PIN, self.ESC_STOP_PULSE_WIDTH)
        self.pi.stop()
        rospy.loginfo("GPIO와 pigpio 리소스를 정리했습니다.")


if __name__ == '__main__':
    controller = MotorController()
    
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.cleanup()
