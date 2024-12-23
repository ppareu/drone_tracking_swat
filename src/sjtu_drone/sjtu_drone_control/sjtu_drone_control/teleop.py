# #!/usr/bin/env python3
# # Copyright 2023 Georg Novotny
# #
# # Licensed under the GNU GENERAL PUBLIC LICENSE, Version 3.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     https://www.gnu.org/licenses/gpl-3.0.en.html
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, Vector3
# from std_msgs.msg import Empty
# import sys
# import termios
# import tty


# MSG = """
# Control Your Drone!
# ---------------------------
# Moving around:
#         w
#     a   s    d
#         x

# t/l: takeoff/land (upper/lower case)
# q/e : increase/decrease linear and angular velocity (upper/lower case)
# A/D: rotate left/right
# r/f : rise/fall (upper/lower case)

# ---------------------------
# CTRL-C to quit
# ---------------------------

# """


# class TeleopNode(Node):
#     def __init__(self) -> None:
#         super().__init__('teleop_node')

#         # Publishers
#         self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
#         self.takeoff_publisher = self.create_publisher(Empty, 'takeoff', 10)
#         self.land_publisher = self.create_publisher(Empty, 'land', 10)

#         # Velocity parameters
#         self.linear_velocity = 0.0
#         self.angular_velocity = 0.0
#         self.linear_increment = 0.05
#         self.angular_increment = 0.05
#         self.max_linear_velocity = 1.0
#         self.max_angular_velocity = 1.0

#         # Start a timer to listen to keyboard inputs
#         self.create_timer((1/30), self.read_keyboard_input)

#     def get_velocity_msg(self) -> str:
#         return "Linear Velocity: " + str(self.linear_velocity) + "\nAngular Velocity: " \
#             + str(self.angular_velocity) + "\n"

#     def read_keyboard_input(self) -> None:
#         """
#         Read keyboard inputs and publish corresponding commands
#         """
#         while rclpy.ok():
#             # Print the instructions
#             print(MSG+self.get_velocity_msg())
#             # Implement a non-blocking keyboard read
#             key = self.get_key()
#             # Handle velocity changes
#             if key.lower() == 'q':
#                 self.linear_velocity = min(self.linear_velocity + self.linear_increment,
#                                            self.max_linear_velocity)
#                 self.angular_velocity = min(self.angular_velocity + self.angular_increment,
#                                             self.max_angular_velocity)
#             elif key.lower() == 'e':
#                 self.linear_velocity = max(self.linear_velocity - self.linear_increment,
#                                            -self.max_linear_velocity)
#                 self.angular_velocity = max(self.angular_velocity - self.angular_increment,
#                                             -self.max_angular_velocity)
#             elif key.lower() == 'w':
#                 # Move forward
#                 linear_vec = Vector3()
#                 linear_vec.x = self.linear_velocity
#                 self.publish_cmd_vel(linear_vec)
#             elif key.lower() == 's':
#                 # Hover
#                 self.publish_cmd_vel()
#             elif key.lower() == 'x':
#                 # Move backward
#                 linear_vec = Vector3()
#                 linear_vec.x = -self.linear_velocity
#                 self.publish_cmd_vel(linear_vec)
#             elif key == 'a':
#                 # Move Left
#                 linear_vec = Vector3()
#                 linear_vec.y = self.linear_velocity
#                 self.publish_cmd_vel(linear_vec)
#             elif key == 'd':
#                 # Move right
#                 linear_vec = Vector3()
#                 linear_vec.y = -self.linear_velocity
#                 self.publish_cmd_vel(linear_vec)
#             elif key == 'A':
#                 # Move Left
#                 angular_vec = Vector3()
#                 angular_vec.z = self.angular_velocity
#                 self.publish_cmd_vel(angular_vec=angular_vec)
#             elif key == 'D':
#                 # Move right
#                 angular_vec = Vector3()
#                 angular_vec.z = -self.angular_velocity
#                 self.publish_cmd_vel(angular_vec=angular_vec)
#             elif key.lower() == 'r':
#                 # Rise
#                 linear_vec = Vector3()
#                 linear_vec.z = self.linear_velocity
#                 self.publish_cmd_vel(linear_vec)
#             elif key.lower() == 'f':
#                 # Fall
#                 linear_vec = Vector3()
#                 linear_vec.z = -self.angular_velocity
#                 self.publish_cmd_vel(linear_vec)
#             # Handle other keys for different movements
#             elif key.lower() == 't':
#                 # Takeoff
#                 self.takeoff_publisher.publish(Empty())
#             elif key.lower() == 'l':
#                 # Land
#                 self.publish_cmd_vel()
#                 self.land_publisher.publish(Empty())

#     def get_key(self) -> str:
#         """
#         Function to capture keyboard input
#         """
#         fd = sys.stdin.fileno()
#         old_settings = termios.tcgetattr(fd)
#         try:
#             tty.setraw(sys.stdin.fileno())
#             ch = sys.stdin.read(1)
#         finally:
#             termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#         return ch

#     def publish_cmd_vel(self, linear_vec: Vector3 = Vector3(),
#                         angular_vec: Vector3 = Vector3()) -> None:
#         """
#         Publish a Twist message to cmd_vel topic
#         """
#         twist = Twist(linear=linear_vec, angular=angular_vec)
#         self.cmd_vel_publisher.publish(twist)


# def main(args=None):
#     rclpy.init(args=args)
#     teleop_node = TeleopNode()
#     rclpy.spin(teleop_node)
#     teleop_node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
#=======================================================================================================================================  Keyboard
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Pose
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from .ui.drone_ctrl import Ui_drone_controller

class DroneNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # Model Loading
        model_path = os.path.join(
            get_package_share_directory("swat_project"),
            "yolov8_model", "best.pt"
        )
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.takeoff_publisher = self.create_publisher(Empty, 'takeoff', 10)
        self.land_publisher = self.create_publisher(Empty, 'land', 10)
        self.nav2_goal_publisher = self.create_publisher(PoseStamped, '/tb1/goal_pose', 10)

        # Subscribers
        self.drone_cam_sub = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.drone_cam_callback,
            10
        )
        self.sub_drone_pose = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self.swat_dispatch_callback, 10) # 드론 좌표

        self.drone_frame = None  # For storing processed frame
        self.tracking_status = False # 트래킹 on/off 상태

    def swat_dispatch_callback(self, drone_location):
        if not self.tracking_status:
            return

        self.drone_position = [drone_location.position.x, drone_location.position.y]
        self.get_logger().info(
            f"Received Drone Pose: x={self.drone_position[0]:.2f}, y={self.drone_position[1]:.2f}"
        )
        try:
            # Nav2에 목표 좌표 전달
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = "map"

            goal_msg.pose.position.x = self.drone_position[0]
            goal_msg.pose.position.y = self.drone_position[1]
            goal_msg.pose.orientation.w = 1.0  # 방향은 기본값 설정

            self.nav2_goal_publisher.publish(goal_msg)
        except Exception as e:
            self.get_logger().error(f"Error while publishing Nav2 goal: {e}")

    def drone_cam_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(source=frame, imgsz=640)
        self.drone_frame = self.draw_bboxes(frame, results)

        if self.tracking_status:
            # 프레임 중심 좌표 계산
            h, w, _ = self.drone_frame.shape
            center_x = w // 2
            center_y = h // 2

            # 트래킹 대상 선택
            best_box = None
            best_conf = 0.0
            for result in results:
                for box in result.boxes:
                    conf = box.conf[0]
                    if conf > 0.80 and conf > best_conf:
                        best_box = box
                        best_conf = conf

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 중심점에 원 그리기 (디버깅용)
                cv2.circle(self.drone_frame, (cx, cy), 5, (0, 255, 0), -1)

                # Δx, Δy 계산
                delta_x = center_x - cx
                delta_y = center_y - cy

                # 이동 명령 계산
                linear_vec = Vector3()
                angular_vec = Vector3()

                # 좌우 회전 설정 (각속도)
                if abs(delta_x) > 50:  # 가로 오차가 클 경우 회전
                    angular_vec.z = 0.5 if delta_x > 0 else -0.5  # 오른쪽/왼쪽 회전

                # 앞뒤 이동 설정 (선형 속도)
                if abs(delta_y) > 50:  # 세로 오차가 클 경우 이동
                    linear_vec.x = 0.5 if delta_y > 0 else -0.5

                self.publish_cmd_vel(linear_vec=linear_vec, angular_vec=angular_vec)

            elif best_box is None:
                self.publish_cmd_vel()  # 드론 정지

    def draw_bboxes(self, frame, results):
        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf > 0.8:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = f'{self.model.names[cls]} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
        return frame

    def publish_cmd_vel(self, linear_vec=Vector3(), angular_vec=Vector3()):
        twist = Twist(linear=linear_vec, angular=angular_vec)
        self.cmd_vel_publisher.publish(twist)


class ROS2Thread(QThread):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def run(self):
        rclpy.spin(self.node)

    def stop(self):
        if self.node:
            self.node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        self.quit()
        self.wait()


class DroneTeleopGUI:
    def __init__(self, ros_node):
        self.node = ros_node
        self.app = QtWidgets.QApplication([])
        self.main_window = QtWidgets.QWidget()
        self.ui = Ui_drone_controller()
        self.ui.setupUi(self.main_window)

        self.drone_cam_label = QtWidgets.QLabel(self.main_window)
        self.drone_cam_label.setGeometry(QtCore.QRect(10, 10, 480, 320))
        self.drone_cam_label.setObjectName("drone_cam_label")
        self.drone_cam_label.setStyleSheet("background-color: black;")
        self.drone_cam_label.setAlignment(QtCore.Qt.AlignCenter)

        # Connect buttons to functions
        self.ui.forward_btn.clicked.connect(self.move_forward)
        self.ui.left_btn.clicked.connect(self.move_left)
        self.ui.right_btn.clicked.connect(self.move_right)
        self.ui.back_btn.clicked.connect(self.move_backward)
        self.ui.stop_btn.clicked.connect(self.hover)
        self.ui.tracking_btn.clicked.connect(self.start_tracking)
        self.ui.speed_up_btn.clicked.connect(self.increase_speed)
        self.ui.speed_down_btn.clicked.connect(self.decrease_speed)
        self.ui.land_btn.clicked.connect(self.land)
        self.ui.tackoff_btn.clicked.connect(self.takeoff)
        self.ui.rotate_left_btn.clicked.connect(self.rotate_left)
        self.ui.rotate_right_btn.clicked.connect(self.rotate_right)
        self.ui.rise_btn.clicked.connect(self.rise)
        self.ui.fall_btn.clicked.connect(self.fall)

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.linear_increment = 0.05
        self.angular_increment = 0.05
        self.max_linear_velocity = 1.0
        self.max_angular_velocity = 1.0

        self.update_lcd_display()

        # ROS2 thread
        self.ros2_thread = ROS2Thread(self.node)
        self.ros2_thread.start()

        # 주기적 화면 갱신을 위한 타이머 (옵션)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(50)  # 50ms 간격으로 update_display() (약 20fps)

    def update_display(self):
        frame = self.node.drone_frame
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_BGR888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            
            self.drone_cam_label.setPixmap(pixmap)
        else:
            # 프레임이 없으면 clear() 등 처리 가능
            self.drone_cam_label.clear()

    def update_lcd_display(self):
        self.ui.linear_lcd.display(round(self.linear_velocity, 2))
        self.ui.angular_lcd.display(round(self.angular_velocity, 2))

    def increase_speed(self):
        self.linear_velocity = min(self.linear_velocity + self.linear_increment, self.max_linear_velocity)
        self.angular_velocity = min(self.angular_velocity + self.angular_increment, self.max_angular_velocity)
        self.update_lcd_display()

    def decrease_speed(self):
        self.linear_velocity = max(self.linear_velocity - self.linear_increment, -self.max_linear_velocity)
        self.angular_velocity = max(self.angular_velocity - self.angular_increment, -self.max_angular_velocity)
        self.update_lcd_display()

    def move_forward(self):
        linear_vec = Vector3(x=self.linear_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def move_backward(self):
        linear_vec = Vector3(x=-self.linear_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def move_left(self):
        linear_vec = Vector3(y=self.linear_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def move_right(self):
        linear_vec = Vector3(y=-self.linear_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def rotate_left(self):
        angular_vec = Vector3(z=self.angular_velocity)
        self.node.publish_cmd_vel(angular_vec=angular_vec)

    def rotate_right(self):
        angular_vec = Vector3(z=-self.angular_velocity)
        self.node.publish_cmd_vel(angular_vec=angular_vec)

    def rise(self):
        linear_vec = Vector3(z=self.linear_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def fall(self):
        linear_vec = Vector3(z=-self.angular_velocity)
        self.node.publish_cmd_vel(linear_vec=linear_vec)

    def takeoff(self):
        self.node.takeoff_publisher.publish(Empty())

    def land(self):
        self.node.publish_cmd_vel()
        self.node.land_publisher.publish(Empty())

    def hover(self):
        self.node.publish_cmd_vel()

    def start_tracking(self):
        if not self.node.tracking_status:
            self.node.tracking_status = True
            self.ui.tracking_btn.setStyleSheet("background-color: red;")
        else:
            self.node.tracking_status = False
            self.ui.tracking_btn.setStyleSheet("") # 버튼 색상 초기화(원래 상태로 되돌림)
            self.node.publish_cmd_vel()

    def run(self):
        self.main_window.show()
        self.app.exec_()
        self.ros2_thread.stop()


def main(args=None):
    rclpy.init(args=args)
    ros_node = DroneNode()
    teleop_gui = DroneTeleopGUI(ros_node)
    teleop_gui.run()

if __name__ == '__main__':
    main()
