import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryActionFeedback , FollowJointTrajectoryGoal , JointTrajectoryControllerState
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist , PoseStamped, Point, Quaternion
import actionlib
from move_base_msgs.msg import MoveBaseActionGoal

#roslaunch tiago_2dnav_gazebo tiago_navigation.launch public_sim:=true

class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self):
        """
        Initializes a new TiagoEnv environment.
        Tiago doesnt use controller_manager, therefore we wont reset the 
        controllers in the standard fashion. For the moment we wont reset them.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /mobile_base_controller/odom : Odometry readings of the Base of the Robot
        * /xtion/depth_registered/image_raw: 2d Depth image of the depth sensor.
        * /xtion/depth_registered/points: Pointcloud sensor readings
        * /xtion/rgb/image_rect_color: RGB camera
        * /scan_raw: Laser Readings
        
        Actuators Topic List: /mobile_base_controller/cmd_vel, 
        
        Args:
        """
        rospy.logdebug("Start TiagoEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        
        # Internal Vars
        self.arm_joints = ['arm_1_joint' , 'arm_2_joint' , 'arm_3_joint' , 'arm_4_joint' , 'arm_5_joint' , 'arm_6_joint' , 'arm_7_joint']
        self.head_joints = ['head_1_joint' , 'head_2_joint']
        self.torso_joints = ['torso_lift_joint']
        self.hand_joints = ['gripper_left_finger_joint' , 'gripper_right_finger_joint']
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            #start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")




        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/mobile_base_controller/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/base_imu", Imu, self._imu_callback)
        #Camera topics
        rospy.Subscriber("/xtion/depth_registered/image_raw", Image, self._camera_depth_image_raw_callback)
        rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, self._camera_depth_points_callback)
        rospy.Subscriber("/xtion/rgb/image_rect_color", Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber("/scan_raw", LaserScan, self._laser_scan_callback)
        #Global Planner
        rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self._plan_callback)
        #Subscribe to joint_state topic
        rospy.Subscriber('/joint_states',JointState,self._joint_state_callback)
        #Arm , Head and Torso topics
        rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, self._arm_state_callback)
        #rospy.Subscriber('/hand_controller/follow_joint_trajectory/feedback', FollowJointTrajectoryActionFeedback, self._hand_feedback_callback)
        #rospy.Subscriber('/head_controller/follow_joint_trajectory/feedback', FollowJointTrajectoryActionFeedback, self._head_feedback_callback)
        #rospy.Subscriber('/torso_controller/follow_joint_trajectory/feedback', FollowJointTrajectoryActionFeedback, self._torso_feedback_callback)


        self._cmd_vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self._goal_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=1)

        #robot controller subscriber
        self._arm_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        #self._torso_pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=1)
        #self._head_pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)
        #self._hand_pub = rospy.Publisher('/hand_controller/command', JointTrajectory, queue_size=1)

        self._check_all_systems_ready()
        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished TiagoEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        # We dont need to check for the moment, takes too long
        #self._check_camera_depth_image_raw_ready()
        #self._check_camera_depth_points_ready()
        #self._check_camera_rgb_image_raw_ready()
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_laser_scan_ready()
        #self._check_plan_ready()
        self._check_joint_state_ready()
        self._check_arm_state_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_arm_state_ready(self):
        self.arm_state = None
        rospy.logdebug("Waiting for /arm_controller/state to be READY...")
        while self.arm_state is None and not rospy.is_shutdown():
            try:
                self.arm_state = rospy.wait_for_message('/arm_controller/state',JointTrajectoryControllerState,timeout = 1.0)
                rospy.logdebug("Current /arm_controller/state READY=>")

            except:
                rospy.logerr("Current /arm_controller/state not ready yet, retrying for getting state")

        return self.arm_state     

    def _check_joint_state_ready(self):
        self.current_joint_positions = None
        rospy.logdebug("Waiting for /joint_states to be READY...")
        while self.current_joint_positions is None and not rospy.is_shutdown():
            try:
                self.current_joint_positions = rospy.wait_for_message('/joint_states',JointState,timeout = 1.0)
                rospy.logdebug("Current /joint_states READY=>")

            except:
                rospy.logerr("Current /joint_states not ready yet, retrying for getting odom")

        return self.current_joint_positions       

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /mobile_base_controller/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/mobile_base_controller/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /mobile_base_controller/odomREADY=>")

            except:
                rospy.logerr("Current /mobile_base_controller/odom not ready yet, retrying for getting odom")

        return self.odom
    
    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /base_imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/base_imu", Imu, timeout=1.0)
                rospy.logdebug("Current /base_imu READY=>")

            except:
                rospy.logerr("Current /base_imu not ready yet, retrying for getting imu")

        return self.imu

        
        
    def _check_camera_depth_image_raw_ready(self):
        self.camera_depth_image_raw = None
        rospy.logdebug("Waiting for /xtion/depth_registered/image_raw to be READY...")
        while self.camera_depth_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_depth_image_raw = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image, timeout=1.0)
                rospy.logdebug("Current /xtion/depth_registered/image_raw READY=>")

            except:
                rospy.logerr("Current /xtion/depth_registered/image_raw not ready yet, retrying for getting camera_depth_image_raw")
        return self.camera_depth_image_raw
        
        
    def _check_camera_depth_points_ready(self):
        self.camera_depth_points = None
        rospy.logdebug("Waiting for /xtion/depth_registered/points to be READY...")
        while self.camera_depth_points is None and not rospy.is_shutdown():
            try:
                self.camera_depth_points = rospy.wait_for_message("/xtion/depth_registered/points", PointCloud2, timeout=1.0)
                rospy.logdebug("Current /xtion/depth_registered/points READY=>")

            except:
                rospy.logerr("Current /xtion/depth_registered/points not ready yet, retrying for getting camera_depth_points")
        return self.camera_depth_points
        
        
    def _check_camera_rgb_image_raw_ready(self):
        self.camera_rgb_image_raw = None
        rospy.logdebug("Waiting for /xtion/rgb/image_rect_color to be READY...")
        while self.camera_rgb_image_raw is None and not rospy.is_shutdown():
            try:
                self.camera_rgb_image_raw = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image, timeout=1.0)
                rospy.logdebug("Current /xtion/rgb/image_rect_color READY=>")

            except:
                rospy.logerr("Current /xtion/rgb/image_rect_color not ready yet, retrying for getting camera_rgb_image_raw")
        return self.camera_rgb_image_raw
        

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan_raw to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                #self.laser_scan = rospy.wait_for_message("/scan_raw", LaserScan, timeout=5.0)
                self.laser_scan = rospy.wait_for_message("/scan_raw", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan_raw READY=>")

            except:
                #rospy.logerr("Current /scan_raw not ready yet, retrying for getting laser_scan")
                rospy.logerr("Current /scan_raw not ready yet, retrying for getting laser_scan")
        return self.laser_scan
    
    def _check_plan_ready(self):
        self.gazebo.unpauseSim()
        self.plan = None
        rospy.logdebug("Waiting for /move_base/GlobalPlanner/plan to be READY...")
        while self.plan is None and not rospy.is_shutdown():
            try:
                self.plan = rospy.wait_for_message("/move_base/GlobalPlanner/plan", Path , timeout=1.0)
                rospy.logdebug("Current /move_base/GlobalPlanner/plan =>")

            except:
                rospy.logerr("Current /move_base/GlobalPlanner/plan not ready yet")
        self.gazebo.pauseSim()
        return self.plan
        

    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data    
    
    def _camera_depth_image_raw_callback(self, data):
        self.camera_depth_image_raw = data
        
    def _camera_depth_points_callback(self, data):
        self.camera_depth_points = data
        
    def _camera_rgb_image_raw_callback(self, data):
        self.camera_rgb_image_raw = data
        
    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _plan_callback(self, data):
        self.plan = data

    def _arm_state_callback(self , data):
        self.arm_state = data
    """
    def _head_feedback_callback(self , data):
        self.head = data 

    def _hand_feedback_callback(self , data):
        self.hand = data 

    def _torso_feedback_callback(self , data):
        self.torso = data    
    """
    def _joint_state_callback(self , data):
        """
        Return position of the robot's arm 
        """
        """
        positions = list(data.position)

        for i, joint_name in enumerate(self.arm_joints):
            if joint_name in data.name:
                idx = data.name.index(joint_name)
                if idx < len(positions):
                    self.current_joint_positions[i] = positions[idx]
        """            
        self.joint_state = data                 
    

        
    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

        while self._goal_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _goal_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_goal_pub Publisher Connected")

        while self._arm_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _arm_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_arm_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def move_arm(self, target_positions, duration=5.0):
        """
        Move the arm to specified joint positions
        
        Args:
            target_positions (list): List of 7 joint positions in radians
            duration (float): Time to complete the movement in seconds

        trajectory_msgs/JointTrajectory msg: 

        std_msgs/Header header
            uint32 seq
            time stamp
            string frame_id
        string[] joint_names
        trajectory_msgs/JointTrajectoryPoint[] points
            float64[] positions
            float64[] velocities
            float64[] accelerations
            float64[] effort
            duration time_from_start    
        """
        self.gazebo.unpauseSim()

        if len(target_positions) != len(self.arm_joints):
            rospy.logerr("Number of positions must match number of joints")
            return False

        # Create trajectory message
        trajectory = JointTrajectory()
        #string[] joint_names
        trajectory.joint_names = self.arm_joints
        
        # Create trajectory point trajectory_msgs/JointTrajectoryPoint[] points
        point = JointTrajectoryPoint()
        # float64[] positions
        point.positions = target_positions
        # float64[] velocities
        point.velocities = [0.0] * len(self.arm_joints)
        # float64[] accelerations
        point.accelerations = [0.0] * len(self.arm_joints)
        #duration time_from_start
        point.time_from_start = rospy.Duration(duration)
        
        trajectory.points.append(point)
        
        # Publish trajectory
        self._arm_pub.publish(trajectory)
        
        # Wait for movement to complete
        rate = rospy.Rate(10)  # 10Hz
        timeout = rospy.Time.now() + rospy.Duration(duration + 1.0)
        
        while rospy.Time.now() < timeout:
            rospy.logdebug("error : " + str(self.movement_completed()))
            if self.movement_completed():
                rospy.logdebug("error : " + str(self.movement_completed()))
                rospy.loginfo("Movement completed successfully")
                self.gazebo.pauseSim()
                return True
            rate.sleep()
        
        rospy.logerr("Movement timed out")
        self.gazebo.pauseSim()
        return False

    def movement_completed(self, tolerance=0.01):
        """
        Check if the movement is completed within tolerance
        """
        if not self.arm_state:
            return False
            
        #current_error = np.array(self.current_joint_positions) - np.array(target_positions)
        rospy.logdebug("error : " + str(np.all(np.abs(np.array(self.arm_state.error.positions)) < tolerance)))
        return np.all(np.abs(np.array(self.arm_state.error.positions)) < tolerance)


    def move_base(self, linear_speed, angular_speed, epsilon=0.05, update_rate=10, min_laser_distance=-1):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        self.gazebo.unpauseSim()

        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Tiago Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
        #time.sleep(0.02)
        '''
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        '''

        self.gazebo.pauseSim()
                        
    
    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate, min_laser_distance=-1):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logwarn("START wait_until_twist_achieved...")
        
        rate = rospy.Rate(update_rate)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05
        
        rospy.logdebug("Desired Twist Cmd>>" + str(cmd_vel_value))
        rospy.logdebug("epsilon>>" + str(epsilon))
        
        linear_speed = cmd_vel_value.linear.x
        angular_speed = cmd_vel_value.angular.z
        
        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon
        angular_speed_plus = angular_speed + epsilon
        angular_speed_minus = angular_speed - epsilon
        
        while not rospy.is_shutdown():
            
            crashed_into_something = self.has_crashed(min_laser_distance)
            
            current_odometry = self._check_odom_ready()
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z
            
            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")
            
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)
            
            if linear_vel_are_close and angular_vel_are_close:
                rospy.logwarn("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            
            if crashed_into_something:
                rospy.logerr("Tiago has crashed, stopping movement!")
                break
            
            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        
        rospy.logwarn("END wait_until_twist_achieved...")
        
        return delta_time
        
    def has_crashed(self, min_laser_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        robot_has_crashed = False
        
        if min_laser_distance != -1:
            laser_data = self.get_laser_scan()
            for i, item in enumerate(laser_data.ranges):
                if item == float ('Inf') or np.isinf(item):
                    pass
                elif np.isnan(item):
                   pass
                else:
                    # Has a Non Infinite or Nan Value
                    if (item < min_laser_distance):
                        rospy.logerr("Tiago HAS CRASHED >>> item=" + str(item)+"< "+str(min_laser_distance))
                        robot_has_crashed = True
                        break
        return robot_has_crashed
    
    def goal_setting(self , x , y , z , yaw):
        #self.gazebo.unpauseSim()

        # Create the goal message
        goal_msg = MoveBaseActionGoal()
    
        # Set the header (with current time and 'map' frame)
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"  
    
        # Set the goal target pose
        goal_msg.goal.target_pose.header.stamp = rospy.Time.now()
        goal_msg.goal.target_pose.header.frame_id = "map"

        # Set the position (x, y) and orientation (yaw converted to quaternion)
        goal_msg.goal.target_pose.pose.position = Point(x, y, z)

        # Convert yaw to quaternion for orientation (only on 2D plane, no roll/pitch)
        goal_msg.goal.target_pose.pose.orientation = Quaternion(0, 0, yaw, 1) 

        # Publish the goal
        rospy.loginfo(f"Publishing goal to /move_base/goal: x={x} , y={y} , z={z} , yaw={yaw}")
        self._goal_pub.publish(goal_msg)

        #self.gazebo.pauseSim()

        

    def get_odom(self):
        return self.odom
    
    def get_imu(self):
        return self.imu
        
    def get_camera_depth_image_raw(self):
        return self.camera_depth_image_raw
        
    def get_camera_depth_points(self):
        return self.camera_depth_points
        
    def get_camera_rgb_image_raw(self):
        return self.camera_rgb_image_raw
        
    def get_laser_scan(self):
        return self.laser_scan
        
    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and 
        
        """
        
