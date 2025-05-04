import rospy
import numpy as np
from gymnasium import spaces
from openai_ros.robot_envs import tiago_env
from gymnasium.envs.registration import register
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import math

max_episode_steps = 100 # Can be any Value

register(
        id='TiagoArmMotion-v0',
        entry_point='openai_ros.task_envs.tiago.tiago_arm_motion:TiagoArm',
        max_episode_steps=max_episode_steps,
    )

class TiagoArm(tiago_env.TiagoEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot3 in the turtlebot3 world
        closed room with columns.
        It will learn how to move around without crashing.
        """

        #define the action space of Tiago's arm
        self.arm_joint_min = [0.07 , -1.50 , -3.46 , -0.32 , -2.07 , -1.39 , -2.07]
        self.arm_joint_max = [2.68 , 1.02 , 1.50 , 2.29 , 2.07 , 1.39 , 2.07]

        self.action_space = spaces.Box(np.array(self.arm_joint_min), np.array(self.arm_joint_max))

        # Only variable needed to be set here
        #self.action_space = spaces.Discrete(number_actions)
        target_position = [0.07 , 0.31 , 0.06 , 0.43 , -1.57 , -0.20 , 0.0]
        rospy.logdebug("Init arm motion")
        self.move_arm(target_position)
        
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
    
        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(TiagoArm, self).__init__()

        

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

    def AprilTagDetection():
        """
        Analyze the environment and found the apriltag
        """    
    
        
        