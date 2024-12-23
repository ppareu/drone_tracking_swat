import sys
import os
import rclpy
from threading import Thread
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import QPixmap, QImage
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from ament_index_python.packages import get_package_share_directory

import yaml
from PIL import Image, ImageDraw

class ROSNode(QThread, Node):
    signal = Signal(list)

    def __init__(self, node_name='ros_subscriber_node'):
        QThread.__init__(self)
        Node.__init__(self, node_name)

        self.amcl_pose_x = 0
        self.amcl_pose_y = 0

        self.dot_size = 2

        # drone
        self.sub_drone_pose = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self.drone_pose_callback, 10)

        # swat
        self.sub_swat_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/tb1/amcl_pose', self.swat_pose_callback, 10)

        # enemy
        self.sub_enemy_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/tb2/amcl_pose', self.enemy_pose_callback, 10)

    def drone_pose_callback(self, msg):
        self.update_pose("drone", msg.position.x, msg.position.y)

    def swat_pose_callback(self, msg):
        self.update_pose("swat", msg.pose.pose.position.x, msg.pose.pose.position.y)

    def enemy_pose_callback(self, msg):
        self.update_pose("enemy", msg.pose.pose.position.x, msg.pose.pose.position.y)

    def update_pose(self, target, x, y):
        self.amcl_pose_x = x
        self.amcl_pose_y = y
        self.signal.emit([target, self.amcl_pose_x, self.amcl_pose_y])
        self.get_logger().info(f'Updated pose: x={x}, y={y}')

    def run(self):
        rclpy.spin(self)


class GUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.ros_node.signal.connect(self.process_signal)

        self.resize = (587, 294)
        self.dot_size = 3

        image_path = os.path.join(
            get_package_share_directory("turtlebot3_multi_robot"),
            "map", "map.pgm"
        )
        yaml_path = os.path.join(
            get_package_share_directory("turtlebot3_multi_robot"),
            "map", "map.yaml"
        )

        image = Image.open(image_path)
        self.width, self.height = image.size
        self.image_rgb = image.convert('RGB')

        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        self.resolution = data['resolution']
        self.map_x = -data['origin'][0]
        self.map_y = data['origin'][1] + self.height * self.resolution
        self.occupied_thresh = data['occupied_thresh']
        self.free_thresh = data['free_thresh']

        image_gray = image.convert('L')
        image_np = np.array(image_gray)
        image_np[image_np >= 255 * self.occupied_thresh] = 255
        image_np[image_np <= 255 * self.free_thresh] = 0
        self.image_rgb = Image.fromarray(np.stack([image_np] * 3, axis=-1), 'RGB')

        self.targets = {}  # 각 대상의 현재 위치 저장
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("ROS2 GUI")
        self.setGeometry(100, 100, 620, 340)

        self.label = QLabel(self)
        self.label.setGeometry(QRect(20, 20, 600, 320))

    def process_signal(self, message):
        target, odom_x, odom_y = message
        self.targets[target] = (self.map_x + odom_x, self.map_y - odom_y)

        print(f"{target} Processed signal: x={self.targets[target][0]}, y={self.targets[target][1]}")
        self.update_image()

    def update_image(self):
        image_copy = self.image_rgb.copy()
        draw = ImageDraw.Draw(image_copy)

        # 저장된 모든 대상의 위치를 순회하며 점 그리기
        for target, (x, y) in self.targets.items():
            if target == "enemy":
                color = "red"
            elif target == "swat":
                color = "blue"
            elif target == "drone":
                color = "green"
            else:
                color = "black"  # 알 수 없는 대상일 경우 기본 색

            draw.ellipse((
                x / self.resolution - self.dot_size,
                y / self.resolution - self.dot_size,
                x / self.resolution + self.dot_size,
                y / self.resolution + self.dot_size),
                fill=color
            )

        image_rotated = image_copy.rotate(90, expand=True)
        image_resized = image_rotated.resize(self.resize)
        pil_image = image_resized.convert('RGBA')
        data = pil_image.tobytes("raw", "RGBA")
        qimage = QImage(data, *self.resize, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)


def main():
    rclpy.init()
    ros_node = ROSNode()

    app = QApplication(sys.argv)
    gui = GUI(ros_node)

    ros_thread = Thread(target=lambda: rclpy.spin(ros_node), daemon=True)
    ros_thread.start()

    gui.show()
    try:
        sys.exit(app.exec_())
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
