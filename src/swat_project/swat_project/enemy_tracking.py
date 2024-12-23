import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os

class EnemyTracking(Node):
    def __init__(self):
        super().__init__('enemy_tracking')

        model_path = os.path.join(
            get_package_share_directory("swat_project"),
            "yolov8_model", "best.pt"
        )
        
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        # drone cam sub
        self.drone_cam_sub = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.drone_cam_callback,
            10
        )

        # drone /cmd_vel pub
        self.drone_ctrl_pub = self.create_publisher(
            Twist,
            '/simple_drone/cmd_vel',
            10
        )

    def drone_cam_callback(self, msg):
        # ROS Image 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO 모델로 객체 감지 수행
        results = self.model.predict(source=frame, imgsz=640)

        # 결과를 처리하여 박스 그리기
        detected_frame = self.draw_bboxes(frame, results)

        # 결과 출력 (디버깅용)
        cv2.imshow('drone cam', detected_frame)
        cv2.waitKey(1)

    def draw_bboxes(self, frame, results):
        # YOLOv8 결과에서 bbox, label, confidence 추출
        for result in results:
            for box in result.boxes:
                conf = box.conf[0]  # 신뢰도
                if conf > 0.9:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                    cls = int(box.cls[0])  # 클래스 ID
                    label = f'{self.model.names[cls]} {conf:.2f}'

                    # 박스와 라벨 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

def main(args=None):
    rclpy.init(args=args)
    node = EnemyTracking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('programe shutdown...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
