#!/usr/bin/env python

import rospy
import paho.mqtt.client as mqtt
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
import json
from cv_bridge import CvBridge
import cv2

class ROSMQTTPublisher:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('mqtt_publisher_node', anonymous=True)

        # MQTT 클라이언트 설정
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect

        # MQTT 서버 연결
        self.mqtt_server = rospy.get_param('~mqtt_server', 'broker.hivemq.com')
        self.mqtt_port = rospy.get_param('~mqtt_port', 1883)
        self.mqtt_topic = rospy.get_param('~mqtt_topic', 'ros_topic')
        self.mqtt_client.connect(self.mqtt_server, self.mqtt_port, 60)

        # ROS 토픽 구독
        self.bridge = CvBridge()
        self.sub_string = rospy.Subscriber('/example_string', String, self.string_callback)
        self.sub_float = rospy.Subscriber('/example_float', Float32, self.float_callback)
        self.sub_image = rospy.Subscriber('/example_image', Image, self.image_callback)

        # MQTT 클라이언트의 루프를 별도의 스레드로 실행
        self.mqtt_client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        rospy.loginfo("Connected to MQTT broker with result code " + str(rc))

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            rospy.logwarn("Unexpected MQTT disconnection. Will auto-reconnect")

    def string_callback(self, msg):
        # String 메시지 처리
        data = {"data": msg.data}
        self.publish_mqtt(data)

    def float_callback(self, msg):
        # Float32 메시지 처리
        data = {"data": msg.data}
        self.publish_mqtt(data)

    def image_callback(self, msg):
        # Image 메시지 처리 (이미지 데이터를 Base64로 인코딩)
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, buffer = cv2.imencode('.jpg', cv_image)
        image_data = buffer.tobytes()
        data = {"image": image_data.hex()}
        self.publish_mqtt(data)

    def publish_mqtt(self, data):
        # JSON 형식으로 MQTT 메시지 송신
        json_data = json.dumps(data)
        self.mqtt_client.publish(self.mqtt_topic, json_data)

    def shutdown(self):
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

if __name__ == '__main__':
    try:
        node = ROSMQTTPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()
