
- 해당 디렉토리 생성
mkdir -p ~/swat_ws

- 해당 디렉토리 이동
cd ~/swat_ws/src

- git clone 실행


- 의존성 설치
rosdep install --from-paths src --ignore-src -r -y

- 빌드하기
colcon build --symlink-install

- 빌드된 패키리 로드 하기
source install/setup.bash

- 실행 명령어 터미널 최소 3개 이상 켜야함
  1. ros2 launch sjtu_drone_bringup sjtu_drone_bringup.launch.py (source install/setup.bash 꼭 하기!!)
  2. ros2 launch turtlebot3_multi_robot gazebo_multi_nav2_world.launch.py (source install/setup.bash 꼭 하기!!)
  3. ros2 run swat_project minimap (source install/setup.bash 꼭 하기!!)
