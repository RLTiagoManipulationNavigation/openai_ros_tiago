import rospy
import numpy as np
from gymnasium import spaces
from openai_ros.robot_envs import tiago_env
from gymnasium.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math
import time
from tf.transformations import quaternion_from_euler
import os
import rospkg
import random
import matplotlib.pyplot as plt

from std_srvs.srv import Empty



max_episode_steps = 150 # Can be any Value

register(
        id='TiagoNavigation-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_navigation:TiagoNav',
        max_episode_steps=100,
    )
class RewardNormalizer:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4  # To avoid division by zero

    def normalize(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var = (self.var * (self.count - 1) + delta * delta2) / self.count
        std = np.sqrt(self.var) + 1e-8
        return reward / std
    
class TiagoNav(tiago_env.TiagoEnv):
    def __init__(self):
        self.reward_normalizer = RewardNormalizer()
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # Goal position
        self.goal_pos = np.array([
                                rospy.get_param('/Test_Goal/x'),
                                rospy.get_param('/Test_Goal/y'),
                                rospy.get_param('/Test_Goal/z')
                                ])
        
        # Goal position  epsilon , maximum error for goal position 
        self.goal_eps = np.array([
                                    rospy.get_param('/Test_Goal/eps_x'),
                                    rospy.get_param('/Test_Goal/eps_y'),
                                    rospy.get_param('/Test_Goal/eps_z')
                                ])
        
        # Actions and Observations
        
        self.new_ranges = rospy.get_param('/Tiago/new_ranges')
        self.max_laser_value = rospy.get_param('/Tiago/max_laser_value')
        self.min_laser_value = rospy.get_param('/Tiago/min_laser_value')
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
        self.min_range = rospy.get_param('/Tiago/min_range') # Minimum meters below wich we consider we have crashed

        self.n_discard_scan = rospy.get_param("/Tiago/remove_scan")


        #reward weights
        self.collision_weight = rospy.get_param("/Reward_param/collision_weight")
        self.guide_weight = rospy.get_param("/Reward_param/guide_weight")
        self.proximity_weight = rospy.get_param("/Reward_param/proximity_weight")
        self.collision_reward = rospy.get_param("/Reward_param/collision_reward")
        self.obstacle_proximity = rospy.get_param("/Reward_param/obstacle_proximity")
        self.distance_weight = rospy.get_param("/Reward_param/distance_weight")

        #training parameter 
        self.single_goal = rospy.get_param("/Training/single_goal")
        self.dyn_path = rospy.get_param("/Training/dyn_path")
        self.waypoint_dist = rospy.get_param("/Training/dist_waypoint")
        self.n_waypoint = rospy.get_param('/Training/n_waypoint')
        self.ahead_dist = rospy.get_param("/Training/ahead_dist")


        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        
        laser_scan = self._check_laser_scan_ready()

        num_laser_readings = int(len(laser_scan.ranges))
        rospy.logdebug("num_laser_readings : " + str(num_laser_readings))
        high = np.full((num_laser_readings), laser_scan.range_max)
        low = np.full((num_laser_readings), laser_scan.range_min)
        
        # Generate observation space
        self.observation_space = spaces.Box(low, high)
    
        # Set possible value of linear velocity and angular velocity 
        min_velocity = [self.min_linear_velocity , self.min_angular_velocity]
        max_velocity = [self.max_linear_velocity , self.max_angular_velocity]

        #Generate action space
        self.action_space = spaces.Box(np.array(min_velocity), np.array(max_velocity))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.curr_robot_pos = np.array([0,0,0])
        #used for control if the robot are block 
        self.max_stationary_step = 0

        
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoNav, self).__init__()
        self.rospack = rospkg.RosPack()
        self.path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , 0.0 , 0.0)

        """with open(self.rospack.get_path('tiago_navigation') + "/data/global_path.txt", 'a') as file:
            # Append the new data to the end of the file
            file.write(str(self.path))"""  

        self.initial_position = self.gazebo_robot_state()
        self.initx = self.initial_position[0]
        self.inity = self.initial_position[1]
        self.initw = self.initial_position[3]
        

    def _set_init_pose(self):
        """
            Sets the Robot in its init pose and reset simulation environment
        """
        #reset simulation
        self.gazebo_reset(self.initx , self.inity , self.initw)
        # Short delay for stability
        rospy.sleep(0.5)
        #random_index = random.randint(0, (len(self.paths) - 1))
        #self.path = self.paths[random_index]
        rospy.loginfo(" Initial position : " + str(self.gazebo_robot_state()))

        rospy.loginfo(str(self.amcl_position()))
        # self.reset_position()
        self.move_base( 0.0,
                        0.0)
        
        
        #self.path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , 0.0 , 0.0)
        #goal = self._initialize_goal_position()
        #self.path = self.goal_setting( goal[0] , goal[1] , 0.0 , 0.0 , 0.0 , 0.0)
        #self.goal_pos = np.array([goal[0] , goal[1] , 0])
        #self.get_laser_scan()
        
        return True

    def _initialize_goal_position(self):
        distance = np.random.uniform(0.1, 1.5)
        prob = np.random.rand()
        if prob <= 0.2:
            # Straight-line movement
            goal_pos = [distance , 0.0]
            #action = np.array([np.cos(direction), np.sin(direction)])
        elif prob <= 0.5 and prob > 0.2:
            # Curvy movement
            direction = np.random.uniform( -np.pi/2, np.pi/2)
            goal_pos = [distance*np.cos(direction) , distance*np.sin(direction)]
        else:
            # Fully random movement
            direction = np.random.uniform(-np.pi, np.pi)
            goal_pos = [distance*np.cos(direction) , distance*np.sin(direction)]
        return goal_pos

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.cumulated_reward = 0.0
        self._episode_done = False
        self.truncated = False
        self.cumulated_steps = 0


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        linear = ((action[0] + 1) * (self.max_linear_velocity - self.min_linear_velocity) / 2 + self.min_linear_velocity)
        angular = ((action[1] + 1) * (self.max_angular_velocity - self.min_angular_velocity) / 2 + self.min_angular_velocity)

        # We tell Tiago the linear and angular speed to set to execute
        self.move_base(linear , angular)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data and position of robot
        laser_scan = self.get_laser_scan()
        base_coord = self.amcl_position()
        #local_costmap = self.gen_local_costamp()
        info = {}

        """if self.dyn_path:
            new_path = self.goal_setting( self.goal_pos[0] , self.goal_pos[1] , 0.0 , 0.0 , base_coord[0] , base_coord[1])
            if new_path is not None:
                self.path = new_path"""
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                    ) 
        waypoints , final_pos = self.find_upcoming_waypoint(self.path , base_coord[:2] , self.n_waypoint , base_coord[3])

        if abs(self.goal_pos[0] - base_coord[0]) < self.goal_eps[0] and abs(self.goal_pos[1] - base_coord[1]) < self.goal_eps[1] :
            self.truncated = True
            rospy.loginfo("Goal reach at position : " + str(base_coord))
            info["goal_reach"] = True  
        #rospy.loginfo(str(final_pos))
        # Insert the waypoints into the dictionary
        info['waypoints'] = waypoints
        info['final_pos'] = []
        info['final_pos'].append(self.convert_global_to_robot_coord( float(rospy.get_param('/Test_Goal/x')) , float(rospy.get_param('/Test_Goal/y')) , base_coord[:2] , base_coord[3]))
        info['curr_pos'] = []
        info['curr_pos'].append(base_coord[:2])
        info['truncated'] = self.truncated
        
        rospy.logdebug("END Get Observation ==>")
        #control of stationary of robot
        
        if np.linalg.norm(self.curr_robot_pos - base_coord[:3]) < 0.001:
            self.max_stationary_step += 1
        else:
            self.curr_robot_pos = base_coord[:3]
            self.max_stationary_step = 0   

        return discretized_observations , info
    
    def find_upcoming_waypoint(self , path_coords , robot_pos , n_waypoint , yaw):
        waypoints = []

        count_waypoint = 1
        
        closest_idx, _ = self.find_closest_waypoint(path_coords, robot_pos)

        if closest_idx != path_coords.shape[0] - 1 :
            curr_waypoint_x , curr_waypoint_y = path_coords[closest_idx+1] 
            # Store result
            waypoints.append(self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw))

            for x_i, y_i in [tuple(coord) for coord in path_coords[closest_idx+1:]]: 
                
                if math.sqrt((x_i - curr_waypoint_x)**2 + (y_i - curr_waypoint_y)**2) >= self.waypoint_dist:
                    #update waypoint
                    curr_waypoint_x = x_i
                    curr_waypoint_y = y_i
                    # Store result
                    waypoints.append(self.convert_global_to_robot_coord(x_i , y_i , robot_pos , yaw))

                    count_waypoint += 1

                    if count_waypoint == self.n_waypoint :
                        break 

        
        # If not enough waypoints were added, fill with the last element of path_coords
        if len(waypoints) < n_waypoint:
            curr_waypoint_x , curr_waypoint_y = path_coords[-1] 
        
            waypoints.extend([self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw)] * (n_waypoint - len(waypoints)))
        #generate coord of terminal position 
        final_pos = []

        curr_waypoint_x , curr_waypoint_y = path_coords[-1] 
        
        final_pos.append(self.convert_global_to_robot_coord(curr_waypoint_x.copy() , curr_waypoint_y.copy() , robot_pos , yaw))

        return waypoints , final_pos
    
    def convert_global_to_robot_coord(self , x_i , y_i , robot_pos , yaw):

        # Translate
        x_prime = x_i - robot_pos[0]
        y_prime = y_i - robot_pos[1]
            
        # Rotate
        x_double_prime = x_prime * math.cos(yaw) + y_prime * math.sin(yaw)
        y_double_prime = -x_prime * math.sin(yaw) + y_prime * math.cos(yaw)

        #if self.norm_input:
            # Convert to simple float values
        x_double_prime = float(x_double_prime)/3.5
        y_double_prime = float(y_double_prime)/3.5

        """if x_double_prime > 1.0:
            x_double_prime = 1.0
        elif x_double_prime < -1.0:
            x_double_prime = -1.0       
        if y_double_prime > 1.0:    
            y_double_prime = 1.0
        elif y_double_prime < -1.0:
            y_double_prime = -1.0"""

        return x_double_prime , y_double_prime


    def _is_done(self, observations):
        
        if min(observations) <= self.min_range :
            rospy.logerr("Tiago is Too Close to wall==>")
            self._episode_done = True    
        
        #control if robot are block
        if self.max_stationary_step == 30:
            self.max_stationary_step = 0
            self._episode_done = True
            rospy.loginfo("Robot are block!")
              

        return self._episode_done

    def _compute_reward(self, observations, done):

        base_coord = self.amcl_position()
        reward = 0

        #RMSE for calclate distance between current and goal position
        distance_error = np.linalg.norm(self.goal_pos - base_coord[:3])
        #reward += -self.distance_weight*distance_error
        
        #collision reward
        if min(observations) < self.min_range :
            reward += self.collision_weight * self.collision_reward

        #proximity reward   
        if min(observations) < self.obstacle_proximity: 
            #collision = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , collision_distance))
            #obstalce_reward = -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations)))
            reward += -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , min(observations)))
        
        #guide reward
        if len(self.path) != 0 :
            #rospy.loginfo("inside guide_weight : " + str(self.guide_weight*self.guide_reward(self.path , base_coord[:2])))
            reward += self.guide_weight*self.guide_reward(self.path , base_coord[:2] , base_coord[3])
           
        #if self.truncated:
        #    reward += 100
        
        # Normalize the reward before returning
        #reward = self.reward_normalizer.normalize(reward)
        #         
        return reward
    
    """def plot_test_aheadpt(self , robot_pos , point_ahead , nearest_waypoint , guide_reward):

        # Split into x and y
        x = self.path[:, 0]
        y = self.path[:, 1]
        rospy.loginfo("robot_pos : " + str(robot_pos) + " point_ahead : " + str(point_ahead) + " nearest_waypoint : " + str(nearest_waypoint) + " guide_reward : " + str(guide_reward))
        # Plot
        plt.plot(x, y, marker='o')  # Line with dots at points
        # Add a red dot for robot position 
        plt.plot(robot_pos[0], robot_pos[1], 'ro')  # 'r' = red, 'o' = circle marker
        #add green dot for ahead point 
        plt.plot(point_ahead[0], point_ahead[1], 'go')
        #add yellow dot for nearest waypoint
        plt.plot(nearest_waypoint[0], nearest_waypoint[1], 'yo')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Plot of Points')
        plt.grid(True)
        plt.show()"""
    
    def guide_reward_debug(self , path_coords, robot_pos , yaw):
        """
        Calculate the reward value that permit to calculate the distance of the robot respect a waypoint into the global path 
        with a distance ahead respect the robot defined inside yaml file.

        Args :
            path_coords : 
        
        """
        if len(path_coords) == 0:
            rospy.logerr("Empty path!")
            return 0.0
        # Find closest waypoint
        closest_idx, min_distance = self.find_closest_waypoint(path_coords, robot_pos)
        # Calculate the goal position to reach 0.6m ahead
        goal_position = self.calculate_goal_position(path_coords , closest_idx , self.ahead_dist)
        #goal_position = self.convert_global_to_robot_coord(goal_position[0] , goal_position[1] , robot_pos , yaw)
        # Calculate reward (negative distance to guidance point)
        #distance_to_guidance = np.linalg.norm(goal_position - robot_pos)
        
        return -np.linalg.norm(goal_position - robot_pos) , goal_position , path_coords[closest_idx]

    def guide_reward(self , path_coords, robot_pos , yaw):
        """
        Calculate the reward value that permit to calculate the distance of the robot respect a waypoint into the global path 
        with a distance ahead respect the robot defined inside yaml file.

        Args :
            path_coords : 
        
        """
        if len(path_coords) == 0:
            rospy.logerr("Empty path!")
            return 0.0
        # Find closest waypoint
        closest_idx, min_distance = self.find_closest_waypoint(path_coords, robot_pos)
        # Calculate the goal position to reach 0.6m ahead
        goal_position = self.calculate_goal_position(path_coords , closest_idx , self.ahead_dist)
        #goal_position = self.convert_global_to_robot_coord(goal_position[0] , goal_position[1] , robot_pos , yaw)
        # Calculate reward (negative distance to guidance point)
        #distance_to_guidance = np.linalg.norm(goal_position - robot_pos)
        
        return -np.linalg.norm(goal_position - robot_pos)


    def find_closest_waypoint(self , path_coords, robot_pos):
        """
        Find the closest waypoint on the path to the robot's position.
        
        """
        path_coords = np.array(path_coords, dtype=np.float32)
        robot_pos = np.array(robot_pos, dtype=np.float32)
        distances = np.linalg.norm(path_coords - robot_pos, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx, distances[closest_idx]
    
    def calculate_goal_position(self , path_coords , closest_idx , cum_distance):
        """
        Calculate cumulative distances along the path.
        
        """

        """accum_dist = 0.0
        last_point = global_path[closest_idx]
        guide_point = last_point  # default in case we don't accumulate enough

        for i in range(closest_idx + 1, len(global_path)):
            delta = np.linalg.norm(global_path[i] - last_point)
            accum_dist += delta
            last_point = global_path[i]
            if accum_dist >= guide_distance:
                guide_point = global_path[i]
                break"""

        ahead_dist = 0.0
        curr_index = closest_idx
        curr_pos = path_coords[closest_idx]

        #control if there is only the final goal position of global path
        if curr_index == path_coords.shape[0] - 1 :
              return curr_pos

        goal_coord = curr_pos

        while ahead_dist < cum_distance :
            curr_index += 1
            goal_coord = path_coords[curr_index]    
            ahead_dist += np.linalg.norm(goal_coord - curr_pos)
            curr_pos = goal_coord
            #control if reach the final position of global path 
            if curr_index == path_coords.shape[0] - 1 :
                break
        return goal_coord
    
    """    def step(self , action):
        # Execute the action
        self._set_action(action)
    
        # Get the new observation
        observation, info = self._get_obs()
    
        # Check if the episode is done
        done = self._is_done(observation)
    
        # Compute the reward
        reward = self._compute_reward(observation, done)
        
        # Return the results
        return observation, reward, done, self.truncated, info"""


    # Internal TaskEnv Methods
    
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        new_data = data.ranges[self.n_discard_scan:-self.n_discard_scan]
        self._episode_done = False
        
        discretized_ranges = []
        #mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("n_elements=" + str(len(new_data)/new_ranges))
        for i, item in enumerate(new_data):
            #if (i%new_ranges==0):
            if item == float ('inf') or np.isinf(item) or (float((item)) > self.max_laser_value):
                discretized_ranges.append(self.max_laser_value)
            elif np.isnan(item) or (float(item) < self.min_laser_value):
                discretized_ranges.append(self.min_laser_value)
            else:
                discretized_ranges.append(float(item))
                """    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                """    
        rospy.logdebug("New observation dimension : " + str(len(discretized_ranges)))
        
        #for make the laser scan value divisible by 90
        discretized_ranges[:2] = [25.0, 25.0]
        discretized_ranges[-2:] = [25.0, 25.0]

        return discretized_ranges
