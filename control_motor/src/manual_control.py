#!/usr/bin/env python3

import rospy
import pigpio
import sys
import threading
import queue

class MotorController:
    def __init__(self):
        rospy.init_node('motor_controller', anonymous=True)

        # pigpio 초기화
        self.pi = pigpio.pi()
        if not self.pi.connected:
            rospy.logerr("pigpio 데몬에 연결할 수 없습니다. 데몬을 실행하고 다시 시도해 주세요.")
            rospy.signal_shutdown("pigpio 데몬에 연결할 수 없음")
            return  # pigpio 초기화 실패 시 종료

        # 초기 상태 변수 설정
        self.motor_pwm = 1500  # 기본 모터 PWM (정지)
        self.servo_pwm = 1480  # 기본 서보 PWM (중앙)

        # GPIO 핀 번호 (BCM 모드)
        self.STEERING_SERVO_PIN = 17  # 서보모터 신호 핀 (조향용)
        self.ESC_PIN = 18             # ESC 신호 핀 (속도 제어용)

        # 서보모터의 안전한 기본 펄스 폭 범위 (마이크로초)
        self.STEERING_MIN_PULSE_WIDTH = 1200
        self.STEERING_MAX_PULSE_WIDTH = 1800
        self.STEERING_CENTER_PULSE_WIDTH = 1480

        # ESC의 안전한 기본 펄스 폭 범위 (마이크로초)
        self.ESC_MIN_PULSE_WIDTH = 1300
        self.ESC_MAX_PULSE_WIDTH = 1700
        self.ESC_STOP_PULSE_WIDTH = 1500

        # PWM 변경 단계 설정
        self.STEERING_STEP = 3  # 서보모터 PWM 변화량
        self.ESC_STEP = 1       # ESC PWM 변화량

        # 입력 큐 초기화
        self.input_queue = queue.Queue()

        # 입력 스레드 시작
        self.input_thread = threading.Thread(target=self.input_listener)
        self.input_thread.daemon = True
        self.input_thread.start()

        rospy.loginfo("Motor Controller Node Initialized")

    def input_listener(self):
        """터미널 입력을 감지하여 큐에 저장하는 스레드"""
        while not rospy.is_shutdown():
            key = sys.stdin.read(1)
            self.input_queue.put(key)

    def process_data(self):
        # 서보모터와 ESC 설정
        self.set_steering_servo_pulsewidth(self.servo_pwm)
        self.set_esc_pulsewidth(self.motor_pwm)

    def set_steering_servo_pulsewidth(self, pulsewidth):
        pulsewidth = max(self.STEERING_MIN_PULSE_WIDTH, min(self.STEERING_MAX_PULSE_WIDTH, pulsewidth))
        rospy.loginfo(f"Setting steering servo pulsewidth: {pulsewidth}")
        self.pi.set_servo_pulsewidth(self.STEERING_SERVO_PIN, pulsewidth)

    def set_esc_pulsewidth(self, pulsewidth):
        pulsewidth = max(self.ESC_MIN_PULSE_WIDTH, min(self.ESC_MAX_PULSE_WIDTH, pulsewidth))
        rospy.loginfo(f"Setting ESC pulsewidth: {pulsewidth}")
        self.pi.set_servo_pulsewidth(self.ESC_PIN, pulsewidth)
    
    def run(self):
        rate = rospy.Rate(15)  # 주사율 15Hz
        while not rospy.is_shutdown():
            self.check_manual_control()  # 키보드 입력 체크
            self.process_data()
            rate.sleep()

    def check_manual_control(self):
        while not self.input_queue.empty():
            key = self.input_queue.get()

            if key == 'w':
                self.motor_pwm += self.ESC_STEP
            elif key == 's':
                self.motor_pwm -= self.ESC_STEP
            elif key == 'a':
                self.servo_pwm -= self.STEERING_STEP
            elif key == 'd':
                self.servo_pwm += self.STEERING_STEP

            # PWM 값이 안전한 범위를 넘지 않도록 클램프
            self.motor_pwm = max(self.ESC_STOP_PULSE_WIDTH, min(self.ESC_MAX_PULSE_WIDTH, self.motor_pwm))
            self.servo_pwm = max(self.STEERING_MIN_PULSE_WIDTH, min(self.STEERING_MAX_PULSE_WIDTH, self.servo_pwm))
        
        # # 아무 키도 눌리지 않은 경우 모터 PWM을 1씩 감소 (하한 1500)
        # self.motor_pwm = max(1500, self.motor_pwm - 1)

    def cleanup(self):
        rospy.loginfo("Cleaning up...")
        self.pi.set_servo_pulsewidth(self.STEERING_SERVO_PIN, self.STEERING_CENTER_PULSE_WIDTH)
        self.pi.set_servo_pulsewidth(self.ESC_PIN, self.ESC_STOP_PULSE_WIDTH)
        self.pi.stop()
        rospy.loginfo("GPIO와 pigpio 리소스를 정리했습니다.")

if __name__ == '__main__':
    controller = MotorController()
    
    try:
        if controller.pi.connected:
            controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        controller.cleanup()
