import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        # 카메라 이미지 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # CvBridge 초기화
        self.br = CvBridge()
        
        # 이미지 저장 디렉토리 생성
        self.image_save_path = 'captured_images'
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)
        self.get_logger().info(f"Images will be saved to: {self.image_save_path}")

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image data...')
        
        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 파일 이름 생성 및 저장
            image_filename = os.path.join(
                self.image_save_path,
                f'image_{self.get_clock().now().to_msg().sec}.jpg'
            )
            cv2.imwrite(image_filename, cv_image)
            self.get_logger().info(f'Image saved to {image_filename}')
        
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        # Ctrl+C로 노드 종료 시 로그 출력
        image_subscriber.get_logger().info('Shutting down...')
    finally:
        cv2.destroyAllWindows()
        image_subscriber.destroy_node()  # 노드를 안전하게 정리
        rclpy.shutdown()  # ROS 2 시스템 종료

if __name__ == '__main__':
    main()
