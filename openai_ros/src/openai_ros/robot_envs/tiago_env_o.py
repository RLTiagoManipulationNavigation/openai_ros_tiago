import numpy
import rospy
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class TiagoEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Superclass of environments.
    """

    def __init__(self):
        """
        Initialize a new tiago environment 

        Sensor Topic List:
        * /mobile_base_controller/odom : Odometry readings of the Base of the Robot
        * /base_imu: Inertial Mesuring Unit that gives relative accelerations and orientations.
        * /scan_raw: Laser Readings
        * /xtion/depth_registered/image_raw : depth image 

        
        Tiago base movement control topic : /mobile_base_controller/cmd_vel,
        Tiago arm joints movement control topic :  
        """

        rospy.logdebug("Start TiagoEnv INIT...")

        #internal variables 
        self.controllers_list = []
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(TiagoEnv, self).__init__(controllers_list=self.controllers_list,
                                        robot_name_space=self.robot_name_space,
                                        reset_controls=False,
                                        start_init_physics_parameters=False)
        
        
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        

        # We Start all the ROS related Subscribers and publishers
        #rospy.Subscriber("/mobile_base_controller/odom", Odometry, self._odom_callback)
        #rospy.Subscriber("/base_imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan_raw", LaserScan, self._laser_scan_callback)
        rospy.Subscriber("/xtion/depth_registered/image_raw" , Image , self._depth_image_callback)

        self._cmd_vel_pub = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=1)

        self._check_all_sensors_ready()

        self._check_publishers_connection()

        #self.gazebo.pauseSim()
        
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
    


    # methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_laser_scan_ready()
        #self._check_odom_ready()
        #self._check_imu_ready()
        self._check_depth_image_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_depth_image_ready(self):
        self.depth_image = None
        rospy.logdebug("Waiting for /xtion/depth_registered/image_raw to be READY...")
        while self.depth_image is None and not rospy.is_shutdown():
            try:
                self.depth_image = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image, timeout=1.0)
                rospy.logdebug("Current /xtion/depth_registered/image_raw READY=>")

            except:
                rospy.logerr("Current /xtion/depth_registered/image_raw not ready yet, retrying for getting depth image")

        return self.depth_image    

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /mobile_base_controller/odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/mobile_base_controller/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /mobile_base_controller/odom READY=>")

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


    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan_raw to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan_raw", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan_raw READY=> ")

            except:
                rospy.logerr("Current /scan_raw not ready yet, retrying for getting laser_scan")
        return self.laser_scan
        

    def _odom_callback(self, data):
        self.odom = data
    
    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _depth_image_callback(self , data):
        self.depth_image = data
        
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

        rospy.logdebug("All Publishers READY")


    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, linear_speed, angular_speed):
                  #, epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return: 
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        #rospy.logdebug("Tiago Cmd>>" + str/imu(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        #self.wait_until_twist_achieved(cmd_vel_value,
        #                                epsilon,
        #                                update_rate)

    #def control_arm_motion():

    '''
    def wait_until_twist_achieved(self, cmd_vel_value, epsilon, update_rate):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rospy.logdebug("START wait_until_twist_achieved...")
        
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
            current_odometry = self._check_odom_ready()
            # IN turtlebot3 the odometry angular readings are inverted, so we have to invert the sign.
            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = -1*current_odometry.twist.twist.angular.z
            
            rospy.logdebug("Linear VEL=" + str(odom_linear_vel) + ", ?RANGE=[" + str(linear_speed_minus) + ","+str(linear_speed_plus)+"]")
            rospy.logdebug("Angular VEL=" + str(odom_angular_vel) + ", ?RANGE=[" + str(angular_speed_minus) + ","+str(angular_speed_plus)+"]")
            
            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (odom_linear_vel > linear_speed_minus)
            angular_vel_are_close = (odom_angular_vel <= angular_speed_plus) and (odom_angular_vel > angular_speed_minus)
            
            if linear_vel_are_close and angular_vel_are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        
        rospy.logdebug("END wait_until_twist_achieved...")
        
        return delta_time
    '''    

    def get_odom(self):
        return self.odom
        
    def get_imu(self):
        return self.imu
        
    def get_laser_scan(self):
        return self.laser_scan  
    
    def get_depth_image(self):
        return self.depth_image  

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