import rospy
import numpy as np
from gymnasium import spaces
from openai_ros.robot_envs import tiago_env
from gymnasium.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

max_episode_steps = 1000000 # Can be any Value

register(
        id='TiagoNavigation-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_navigation:TiagoNav',
        max_episode_steps=max_episode_steps,
    )

class TiagoNav(tiago_env.TiagoEnv):
    def __init__(self):
        """
        This Task Env is designed for Tiago navigation task
        Learn how reach the goal avoiding the static and dynamic obstacles
        """
        
        # Set action space basend on value of linear and angular velocity
        low_speed = np.array([-1.5, -1.5], dtype=np.float32)
        high_speed = np.array([1.5, 1.5], dtype=np.float32)
        self.action_space = spaces.Box(low_speed,
                                    high_speed,
                                    dtype=np.float32)
        
        # We set the reward range.
        self.reward_range = (-np.inf, np.inf)
        
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()
        #rospy.logdebug("Angle max : " + str(laser_scan.angle_max) + ", angle min :" + str(laser_scan.angle_min) + " , angle increment : " + str(laser_scan.angle_increment))
        high = np.full((len(laser_scan.ranges)), laser_scan.range_max)
        low = np.full((len(laser_scan.ranges)), laser_scan.range_min)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.collision_weight = rospy.get_param("/Reward_param/collision_weight")
        self.guide_weight = rospy.get_param("/Reward_param/guide_weight")
        self.proximity_weight = rospy.get_param("/Reward_param/proximity_weight")
        self.proximity_bound = rospy.get_param("/Reward_param/obstacle_proximity")
        self.collision_reward = rospy.get_param("/Reward_param/collision_reward")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoNav, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        '''
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        epsilon=0.05,
                        update_rate=10)
        '''
        self.move_base(0,
                       0)
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


    def _set_action(self, angular_speed , linear_speed):
        """
        This set action will Set the linear and angular speed of the tiago
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==> ("+str(linear_speed)+" , "+str(angular_speed))
        
        #self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        self.move_base(linear_speed, angular_speed)
        
        #rospy.logdebug("End Set Action ==> ("+str(linear_speed)+" , "+str(angular_speed))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        laser_scan = self.get_laser_scan()
        
        discretized_observations = self.discretize_scan_observation(    laser_scan,
                                                                        self.new_ranges
                                                                        )

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        return discretized_observations
        
    
    def _is_done(self, observations):
        ###############################################
        #understad how we need to insert in this method
        ############################################### 
        return True
    

    def _compute_reward(self):

        collision_dist = self.check_collision()
        reward = 0
        #collision reward
        if collision_dist < 0.1 :
            reward += self.collision_weight * self.collision_reward
        #proximity reward    
        reward += -1*abs( self.collision_reward - min(self.collision_reward , collision_dist))
        #guide rereward
        reward += 0
        #update total reward
        self.cumulated_reward += reward
        self.cumulated_steps += 1    
        return reward


    # Internal TaskEnv Methods
    
    def discretize_scan_observation(self,data,new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        
        discretized_ranges = []
        mod = len(data.ranges)/new_ranges
        
        rospy.logdebug("data=" + str(data))
        rospy.logdebug("new_ranges=" + str(new_ranges))
        rospy.logdebug("mod=" + str(mod))
        
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    discretized_ranges.append(self.max_laser_value)
                elif np.isnan(item):
                    discretized_ranges.append(self.min_laser_value)
                else:
                    discretized_ranges.append(int(item))
                    
                if (self.min_range > item > 0):
                    rospy.logerr("done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    self._episode_done = True
                else:
                    rospy.logdebug("NOT done Validation >>> item=" + str(item)+"< "+str(self.min_range))
                    

        return discretized_ranges

    def check_collision( self):
        laser_msg = self.get_laser_scan()
        #get minimum range of current laser scan 
        min_range = min(laser_msg.ranges)
        return min_range
        #return -1*abs( self.collision_reward - min(self.collision_reward , min_range))