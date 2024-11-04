import numpy as np

def guide_reward(path_coords, robot_pos):
        """
        Calculate the natural guidance reward.
        
        Args:
            path_coords (np.ndarray): Array of shape (n, 2) containing path x,y coordinates
            robot_pos (np.ndarray): Robot's current position [x, y]
            
        Returns:
            float: Reward value
        """

        # Find closest waypoint
        closest_idx, min_distance = find_closest_waypoint(path_coords, robot_pos)
        
        # Calculate the goal position to reach 0.6m ahead
        ahead_dist = 0.6
        goal_postion = calculate_goal_position(path_coords , closest_idx , ahead_dist)
        
        # Calculate reward (negative distance to guidance point)
        distance_to_guidance = -np.linalg.norm(goal_postion - robot_pos)
        
        
        return distance_to_guidance

def find_closest_waypoint( path_coords, robot_pos):
        """
        Find the closest waypoint on the path to the robot's position.
        
        Args:
            path_coords (np.ndarray): Array of shape (n, 2) containing path x,y coordinates
            robot_pos (np.ndarray): Robot's current position [x, y]
            
        Returns:
            tuple: (closest point index, distance to closest point)
        """
        distances = np.linalg.norm(path_coords - robot_pos, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx, distances[closest_idx]
    
def calculate_goal_position(path_coords , closest_idx , cum_distance):
        """
        Calculate cumulative distances along the path.
        
            
        Returns:
            np.ndarray: Array with the goal position to reach 
        """
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