## Installation 
1. Install ROS and Gazebo, and other linking libraries. 
2. cd catkin_ws
3. catkin_make
3. . devel/setup.bash
4. roscd collision_avoidance_env/scripts/
5. sudo pip install -e .
6. roscd collision_avoidance/scripts
7. sudo pip install -r requirements.txt

The first time you open Gazebo, it will download all models from the Gazebo servers, which may take some time. Run rosrun gazebo_ros gazebo to run Gazebo and install models.

## Command
1. roscd collision_avoidance/scripts
2. python main.py
