#!/usr/bin/env python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import rospy
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import Imu, Image
from gazebo_msgs.msg import ModelStates, ModelState, ContactsState
from gazebo_msgs.srv import SetModelState

from cv_bridge import CvBridge, CvBridgeError

import numpy as np

import os
import math
import signal
import subprocess
import time
import random
from os import path

class CollisionEnv(gym.Env):
        metadata = {'render.modes': ['human']}
        def __init__(self):
            # Init ROS Node
            rospy.init_node('gazebo_collision_car_gym')
            
            # Launch the Gazebo simulator with the RC car
            self.gazeboProcess = subprocess.Popen(["roslaunch", "collision_avoidance_gazebo", "avoidance.launch"])
            time.sleep(10)
            # Launch the control system to control the RC car
            self.controlProcess = subprocess.Popen(["roslaunch", "collision_avoidance_gazebo_control", "rc_car_control.launch"])
            time.sleep(5)
                            
            print ("Gazebo launched!")      
            
            self.gzclient_pid = 0
            # Get the ROS publishers to the steering and throttle 
            self.throtle1 = rospy.Publisher('/rc_car/left_rear_axle_controller/command', Float64, queue_size = 1)
            self.throtle2 = rospy.Publisher('/rc_car/right_rear_axle_controller/command', Float64, queue_size = 1)
            self.steer1 = rospy.Publisher('/rc_car/left_steering_joint_controller/command', Float64, queue_size = 1)
            self.steer2 = rospy.Publisher('/rc_car/right_steering_joint_controller/command', Float64, queue_size = 1)
    
            # Control services for Gazebo
            self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            
            # Reward related
            self.reward_range = (-np.inf, np.inf)
            self.last_pose = {"x": 0, "y": 0} 

            # Action related
            carDegs = 25
            self.radianMappings = [math.radians(x-90) for x in range(90 - carDegs, 90 + carDegs+1, 10)]
            self.action_space = spaces.Discrete(len(self.radianMappings))
            self.throttle = 1490       
            
            # State related
            self.image_size = 84 * 84 * 3
            high = np.ones(self.image_size) * 255
            self.observation_space = spaces.Box(np.zeros(self.image_size), high)   
            self.bridge = CvBridge()
            
            self._seed()
                
        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed] 
        
        def applyThrottle(self, throtle):
            # Apply Throttle by publishing a message to the throttle publishers
            self.throtle1.publish(throtle)
            self.throtle2.publish(throtle)

        def applySteering(self, steering):
            # Apply Throttle by publishing a message to the steering publishers
            self.steer1.publish(steering)
            self.steer2.publish(steering)

        def _step(self, action):
            # Unpause the Gazebo simulator
            self.unpausePhysics()

            # Take the action
            if isinstance(action, tuple):
                    self.applyThrottle(action[0])
                    action = action[1]
            else:
                    self.applyThrottle(self.throttle)
            
            self.applySteering(self.radianMappings[action])

            # Get the new state
            image = self.getImageData()
            posData = self.getPosData()
            isColliding = self.getCollisionData()
            self.pausePhysics()

            done = self.isDone(isColliding, posData)

            # Calculate reward as the distance moved since last step; if done then a collision has been detected so give -50 penalty
            x = posData.pose[-1].position.x
            y = posData.pose[-1].position.y
            distance_since_last_step = math.sqrt((self.last_pose["x"] - x) ** 2 + (self.last_pose["y"] - y) ** 2)
            self.last_pose = {"x": x, "y": y}
            reward = distance_since_last_step if not done else -50
            
            return np.array(image), reward, done, {}

        def isDone(self, isColliding, posData):
            # Done is True if a collision has been detected; as a fail safe, a collision is also 'detected' if the car stops moving i.e. velocity is ~0    
            x = posData.pose[-1].position.x
            y = posData.pose[-1].position.y
            distance_from_origin = math.sqrt(x ** 2 + y ** 2)

            velx = posData.twist[-1].linear.x
            vely = posData.twist[-1].linear.y
            combined = math.sqrt(velx ** 2 + vely ** 2)

            # Sometimes when the car starts at the (0,0) mark, it takes time to build up speed. Hence, done can only be true after moving 0.1m from origin
            return (isColliding or combined < 0.04) and distance_from_origin > 0.1
                
        def _reset(self):
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                # Reset position of the rc Car
                reset_pose = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                nullPosition = ModelState()
                nullPosition.model_name = "rc_car"
                nullPosition.pose.position.x = 0
                nullPosition.pose.position.y = 0
                nullPosition.pose.position.z = 0.05
                reset_pose(nullPosition)
            except (rospy.ServiceException) as e:
                print ("/gazebo/set_model_state service call failed")
                        
            # Unpause physics and move the car a little, then return the new init state 
            self.unpausePhysics()
            self.applyThrottle(self.throttle)
            self.applySteering(0)
            time.sleep(0.8)
            image = self.getImageData()
            self.pausePhysics()
            
            self.last_pose = {"x": 0, "y": 0}

            return np.array(image)
        
        # Open Gzclient, which is the GUI for Gazebo
        def _render(self, mode='human', close=False):
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount < 1:
                subprocess.Popen("gzclient")
                self.gzclient_pid = int(subprocess.check_output(["pidof","-s","gzclient"]))
            else:
                self.gzclient_pid = 0
    
        # To handle Gazebo crashes gracefully . . .
        def handleGazeboFailure(self):
            print("Failed too many times, trying to restart Gazebo")
            tmp = os.popen("ps -Af").read()
            gzserver_count = tmp.count('gzserver')
            gzclient_count = tmp.count('gzclient')
            control_count = tmp.count('/usr/bin/python /opt/ros/kinetic/bin/roslaunch collision_avoidance_gazebo_control rc_car_control.launch')               
            
            # . . . all the Gazebo related processes are killed . . .
            if gzclient_count > 0:
                os.system("killall -9 gzclient")
            if gzserver_count > 0:
                os.system("killall -9 gzserver")    
            if control_count > 0:
                os.system('pkill -TERM -P {pid}'.format(pid=self.controlProcess.pid))
            
            if (gzclient_count or gzserver_count or control_count > 0):
                os.wait()
                    
            # . . and restarted
            self.gazeboProcess = subprocess.Popen(["roslaunch", "collision_avoidance_gazebo", "avoidance.launch"])
            time.sleep(10)
            self.controlProcess = subprocess.Popen(["roslaunch", "collision_avoidance_gazebo_control", "rc_car_control.launch"])
            time.sleep(5)
    
        def getPosData(self):
            # Fetch position data of the car.. If can't fetch after 10 tries, Gazebo has most probably crashed, so gracefully handle the failure
            failureCount = 0
            posData = None        
            while posData is None:
                    try:
                            posData = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1)
                    except Exception as e:
                            failureCount += 1
                            if failureCount % 10 == 0:
                                    self.handleGazeboFailure()          
                            print(e)
                            pass
            #print("Fetched Pos Data")

        def getCollisionData(self):
            # Fetch contact  data of all possible places the car could be colliding i.e. the wheels and the chassis OR handle crash gracfully as in getPosData()
            failureCount = 0
            collisionCount = 0
            left_front = None
            right_front = None
            left_rear = None
            right_rear = None
            chassis = None         
            while left_front is None or right_front is None or left_rear is None or right_rear is None:
                    try:
                            left_front = rospy.wait_for_message('/rc_car/collision/left_front', ContactsState, timeout=1)
                            right_front = rospy.wait_for_message('/rc_car/collision/right_front', ContactsState, timeout=1)
                            left_rear = rospy.wait_for_message('/rc_car/collision/left_rear', ContactsState, timeout=1)
                            right_rear = rospy.wait_for_message('/rc_car/collision/right_rear', ContactsState, timeout=1)
                            chassis = rospy.wait_for_message('/rc_car/collision/chassis', ContactsState, timeout=1)
                    except Exception as e:
                            failureCount += 1
                            if failureCount % 10 == 0:
                                    self.handleGazeboFailure()          
                            print(e)
                            pass
            
            # Chassis doesn't have contact with the ground, so the collision count should be 0 for a non-colliding chassis
            collisionCount = 0
            try:
                    for contact in chassis.states:
                            if not contact.collision2_name == "ground_plane::link::collision":
                                    collisionCount += 1
            except Exception as e:
                    pass

            # Car is said to be colliding with something if any of the wheels detect a non-ground collision, or chassis collision count is more than 0
            return self.isColliding(left_front) or self.isColliding(right_front) or self.isColliding(left_rear) or self.isColliding(right_rear) or collisionCount > 0 

        def isColliding(self, collisionMsg):
            # All wheels need to be touching the ground.. If not touching ground, it is assumed that a collision has knocked the car into the air
            groundCollision = 0
            collisionCount = 0
            try:
                    for contact in collisionMsg.states:
                            if not contact.collision2_name == "ground_plane::link::collision":
                                    collisionCount += 1
                            else:
                                    groundCollision += 1
                    # Thus, if not touching ground or if other collisions are more than 0, the wheel is said be colliding with something
                    return groundCollision == 0 or collisionCount > 0 
            except Exception as e:
                    return False

        def getImageData(self):
            failureCount = 0
            imageData = None        
            while imageData is None:
                    try:
                        imageData = rospy.wait_for_message('/rc_car/camera/image_raw', Image, timeout=1)
                    except Exception as e:
                            failureCount += 1
                            if failureCount % 10 == 0:
                                    self.handleGazeboFailure()          
                            print(e)
                            pass

            # After fetching the image data, it is converted into a format readable by Numpy with the CV2 library.
            try:
                cv_image = self.bridge.imgmsg_to_cv2(imageData, "rgb8")
            except CvBridgeError as e:
                print(e)
            return np.reshape(cv_image,[self.image_size])

        # Following are basic methods to call a service to perform an action (as state by the function name)
        def pausePhysics(self): 
            rospy.wait_for_service('/gazebo/pause_physics')
            try:
                self.pause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/pause_physics service call failed")
                
        def unpausePhysics(self):
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print ("/gazebo/unpause_physics service call failed")
                    
        def resetSimulation(self):
            rospy.wait_for_service('/gazebo/reset_simulation')
            try:
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print ("/gazebo/reset_simulation service call failed")
    
        # Kill all Gazebo related processes before quitting
        def _close(self):
            tmp = os.popen("ps -Af").read()
            gzclient_count = tmp.count('gzclient')
            gzserver_count = tmp.count('gzserver')
            roslaunch_count = tmp.count('roslaunch')

            if gzclient_count > 0:
                os.system("killall -9 gzclient")
            if gzserver_count > 0:
                os.system("killall -9 gzserver")
            if roslaunch_count > 0:
                os.system("killall -9 roslaunch")

            if (gzclient_count or gzserver_count or roslaunch_count):
                os.wait()