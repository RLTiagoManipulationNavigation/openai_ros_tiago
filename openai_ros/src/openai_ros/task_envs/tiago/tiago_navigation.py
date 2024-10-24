import rospy
import numpy as np
from gymnasium import spaces
from openai_ros.robot_envs import tiago_env
from gymnasium.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math

max_episode_steps = 1000000 # Can be any Value

register(
        id='TiagoNavigation-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_navigation:TiagoNav',
        max_episode_steps=max_episode_steps,
    )

class TiagoNav(tiago_env.TiagoEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """

        # Only variable needed to be set here
        #self.action_space = spaces.Discrete(number_actions)
        
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        self.truncated = False
        #number_observations = rospy.get_param('/turtlebot3/n_observations')
        '''
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        '''
        # Goal position
        self.x = rospy.get_param('/Test_Goal/x')
        self.y = rospy.get_param('/Test_Goal/y')
        self.z = rospy.get_param('/Test_Goal/z')
        self.yaw = 0
        
        # Actions and Observations
        self.init_linear_forward_speed = rospy.get_param('/Tiago/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/Tiago/init_linear_turn_speed')
        
        self.new_ranges = rospy.get_param('/Tiago/new_ranges')
        self.max_laser_value = rospy.get_param('/Tiago/max_laser_value')
        self.min_laser_value = rospy.get_param('/Tiago/min_laser_value')
        self.max_linear_velocity = rospy.get_param('/Tiago/max_linear_velocity')
        self.min_linear_velocity = rospy.get_param('/Tiago/min_linear_velocity')
        self.max_angular_velocity = rospy.get_param('/Tiago/max_angular_velocity')
        self.min_angular_velocity = rospy.get_param('/Tiago/min_angular_velocity')
        self.min_range = rospy.get_param('/Tiago/min_range')


        #reward weights
        self.collision_weight = rospy.get_param("/Reward_param/collision_weight")
        self.guide_weight = rospy.get_param("/Reward_param/guide_weight")
        self.proximity_weight = rospy.get_param("/Reward_param/proximity_weight")
        self.collision_reward = rospy.get_param("/Reward_param/collision_reward")
        self.obstacle_proximity = rospy.get_param("/Reward_param/obstacle_proximity")

        self.distance_weight = rospy.get_param("/Reward_param/distance_weight")
        
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()

        num_laser_readings = int(len(laser_scan.ranges)/self.new_ranges)
        rospy.logdebug("num_laser_readings : " + str(num_laser_readings))
        high = np.full((num_laser_readings), laser_scan.range_max)
        low = np.full((num_laser_readings), laser_scan.range_min)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high)

        # Set possible value of linear velocity and angular velocity 
        min_velocity = [self.min_linear_velocity , self.min_angular_velocity]
        max_velocity = [self.max_linear_velocity , self.max_angular_velocity]
        self.action_space = spaces.Box(np.array(min_velocity), np.array(max_velocity))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
    
        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoNav, self).__init__()
        
        

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)
        #self.goal_setting(self.x , self.y , self.z , self.yaw)
        #rospy.logdebug("path :"+str(self._check_plan_ready()))
        
        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>"+str(action))
        if action[0] >= self.min_linear_velocity and action[0] <= self.max_linear_velocity :
            curr_linear_vel = action[0]
        else :
            if action[0] < self.min_linear_velocity :
                curr_linear_vel = self.min_linear_velocity
            else : 
                curr_linear_vel = self.max_linear_velocity 

        if action[1] >= self.min_angular_velocity and action[1] <= self.max_angular_velocity :
            curr_angular_vel = action[1]
        else :
            if action[1] < self.min_angular_velocity :
                curr_angular_vel = self.min_angular_velocity
            else : 
                curr_angular_vel = self.max_angular_velocity               

        # We tell Tiago the linear and angular speed to set to execute
        self.move_base(curr_linear_vel, curr_angular_vel, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        #necessary in gymnasium reset()
        info = {}
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        #rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
    
        return discretized_observations , info
        
    
    def _is_done(self, observations):
        
        """
        if self._episode_done:
            rospy.logerr("Tiago is Too Close to wall==>")
        else:
            rospy.logwarn("Tiago is NOT close to a wall ==>")
        """
        if min(observations) <= self.min_range :
            rospy.logerr("Tiago is Too Close to wall==>")
            self._episode_done = True    
            
        # Now we check if it has crashed based on the imu
        """
        imu_data = self.get_imu()
        linear_acceleration_magnitude = self.get_vector_magnitude(imu_data.linear_acceleration)
        if linear_acceleration_magnitude > self.max_linear_aceleration:
            rospy.logerr("Tiago Crashed==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
            self._episode_done = True
        else:
            rospy.logerr("DIDNT crash Tiago ==>"+str(linear_acceleration_magnitude)+">"+str(self.max_linear_aceleration))
        """

        return self._episode_done
    '''
    def is_terminated(self , observations):
        return True
    
    def is_truncated(self):
        return True
    '''
    def _compute_reward(self, observations, done):

        odom_data = self.get_odom()
        laser_msg = self.get_laser_scan()
        collision_distance = min(laser_msg.ranges)

        reward = 0

        #RMSE for calclate distance between current and goal position
        distance_error = math.sqrt((self.x - odom_data.pose.pose.position.x)**2 
                                   + (self.y - odom_data.pose.pose.position.y)**2 
                                   + (self.z - odom_data.pose.pose.position.z)**2)
        reward += -self.distance_weight * distance_error
        #collision reward
        if collision_distance < 0.07 :
            reward += self.collision_weight * self.collision_reward
        #proximity reward   
        if collision_distance < 0.1: 
            reward += -self.proximity_weight*abs( self.obstacle_proximity - min(self.obstacle_proximity , collision_distance))
        #guide reward
        
        '''
        if not done:
            if self.last_action == "FORWARDS":
                reward = self.forwards_reward
            else:
                reward = self.turn_reward
        else:
            reward = -1*self.end_episode_points
        '''

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward
    
    def step(self , action):
         # Execute the action
        self._set_action(action)
    
        # Get the new observation
        observation, info = self._get_obs()
    
        # Check if the episode is done
        done = self._is_done(observation)
    
        # Compute the reward
        reward = self._compute_reward(observation, done)
    
        # Return the results
        return observation, reward, done, self.truncated, info


    # Internal TaskEnv Methods
    
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        new_data = data.ranges[21:-21]
        self._episode_done = False
        
        discretized_ranges = []
        #mod = len(data.ranges)/new_ranges
        
        #rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("n_elements=" + str(len(new_data)/new_ranges))
        
        for i, item in enumerate(new_data):
            if (i%new_ranges==0):
                if item == float ('inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
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

        return discretized_ranges
        
        
    def get_vector_magnitude(self, vector):
        """
        It calculated the magnitude of the Vector3 given.
        This is usefull for reading imu accelerations and knowing if there has been 
        a crash
        :return:
        """
        contact_force_np = np.array((vector.x, vector.y, vector.z))
        force_magnitude = np.linalg.norm(contact_force_np)

        return force_magnitude

