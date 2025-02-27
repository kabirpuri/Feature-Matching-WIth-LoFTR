import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

#Student imports
from lib.IK_position_null import IK
from lib.calculateFK import FK

def drop_block(distance=0.09, force=10):
    arm.exec_gripper_cmd(0.09, 10)

def grab_block(distance=0.048, force=52):
    arm.exec_gripper_cmd(0.048, 52)

def calculate_q_via_ik(pos, q_start):
    q_end, rollout, success, message = ik.inverse(pos, q_start, method='J_pseudo', alpha = 0.5)
    
    
    return q_end
    
def comp_filter(curr_reading, prev_reading, alpha):
    filt_pose = (alpha*curr_reading) + ((1 - alpha)*prev_reading)
    return filt_pose

def get_block_world(q):

    alpha=0.80
        
    H_ee_camera = detector.get_H_ee_camera() # Camera in terms of end_effector
    H_c_ee = H_ee_camera 
    _,H_ee_w = fk.forward(q) # End-Effector in terms of world
    H_c_w = H_ee_w @ H_c_ee # Camera in terms of world
    rospy.sleep(2)
    
    for i in range(5):
            
        b_reading_1 = []
        b_reading_2 = []
        
        # detector.get_detections() = Block in terms of camera
        block_det = detector.get_detections()
            
        for (name_1, pose_1) in block_det:
            b_reading_1.append(pose_1)
            print(name_1,'\n',pose_1)
        b_reading_1 = np.array(b_reading_1)

        for (name_2, pose_2) in block_det:
            b_reading_2.append(pose_2)
            print(name_2,'\n',pose_2)
        b_reading_2 = np.array(b_reading_2)
        # print(block_pose2.shape)

        b_reading_comp = comp_filter(b_reading_2, b_reading_2, alpha)
        b_reading_1 = b_reading_2
        b_reading_2 = b_reading_comp

        block_pose_world=[] # Block in terms of world
    for i in range(b_reading_comp.shape[0]):
        block_pose_world.append(H_c_w @ b_reading_comp[i])

        # Returns Block in terms of world
    
    return len(block_pose_world),block_pose_world


def swap_columns(matrix, col1, col2):
        """
        
        """
        #matrix[:, col1], matrix[:, col2] = matrix[:, col2], matrix[:, col1]
        matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
        
        # Debug print to show matrix after swap
        #print("Matrix after swap:")
        #print(matrix)
        
        return matrix


def rotation_matrix_to_angle_axis(rotation_matrix):
        """
        
        """
        # Extract the 3x3 rotation matrix
        rotation = rotation_matrix[:3, :3]
        abs_rotation = np.abs(rotation)
    
    # Find the column closest to [0, 0, 1]
        target_column = np.array([0, 0, 1])
        min_diff = 0.2  
        col_to_swap = -1
    
     # Loop through each column to find the one closest to [0, 0, 1]
        for i in range(3):
        # Calculate the absolute Euclidean distance to [0, 0, 1]
            diff = np.linalg.norm(np.abs(abs_rotation[:, i] - target_column))  # Absolute distance to [0, 0, 1]
        
        # If this column is the closest, update the swap index
            if diff < min_diff:
                min_diff = diff
                col_to_swap = i
                print("Col to swap")
                print(i)
                
          # Debug: Which column to swap
            #print(f"Column {col_to_swap} is closest to [0, 0, 1] and will be swapped with column 2.")
    
    
    # Swap the column that is closest to [0, 0, 1] with the 3rd column
        rotation = swap_columns(rotation, col_to_swap, 2)
        print("Adjusted rotation = ")
        print(rotation)
    
    # Calculate the rotation angle about the z-axis using atan2
        rz = np.arctan2(rotation[1, 0], rotation[0, 0])
        
    
    # Optionally adjust the angle to stay within [-pi, pi]
        if rz > np.pi / 4:
            rz = rz - np.pi / 2
            #print(rz)
        elif rz < -np.pi / 4:
            rz = rz + np.pi / 2
            #print(rz)
        print("Calculated rz - ")
        print(rz)
        
        return rz

# def rotation_matrix_to_angle_axis(R):

#         assert R.shape == (3, 3)

#         axis = 0
#         angsin = 0
#         angcos = 0
#         for i in range(3):
#             if np.isclose(R[2,i], 1, 1e-04):
#                 axis = i
#         if axis ==0:
#             angcos = R[0,1]
#             angsin = R[0,2]
#         if axis ==1:
#             angcos = R[0,0]
#             angsin = R[0,2]
#         if axis ==2:
#             angcos = R[0,0]
#             angsin = R[0,1]
            
            
#         angle = np.arctan2(angsin,angcos)
#         while angle > 2.897 or angle < -2.896:
#             if angle > 2.897:
#                 angle -= pi/2
#             if angle < -2.896:
#                 angle +=pi/2
    
#         return angle

def move_to_static_block(block, q_current):

        ee_rot = np.array(([1,0,0],
	    			[0,-1,0], 
	    			[0,0,-1],
	    			[0,0,0]))
        block_pos = block[:,3]
        block_pos = block_pos.reshape(4,1)
        ee_goal = np.hstack((ee_rot,block_pos))

        angle = rotation_matrix_to_angle_axis(block[:3,:3])       

        ee_align = deepcopy(ee_goal)
        ee_align[2, 3] = 0.4
        
        q_align = calculate_q_via_ik(ee_align, q_current)
        if q_align is not None:
            q_align[-1] = q_align[-1] - angle

        q_block = calculate_q_via_ik(ee_goal, q_align)
        if q_block is not None:
            q_block[-1] = q_block[-1] - angle 

        return q_align, q_block

def set_static_view(q_current):

    if team == 'red':
        pos_above_pickup = np.array(([1, 0, 0, 0.52 ],
                                        [0,-1, 0, -0.2 ], 
                                        [0, 0,-1, 0.52 ], 
                                        [0, 0, 0, 1    ]))

        pos_above_drop = np.array(([1, 0, 0, 0.52 ],
                                    [0,-1, 0, 0.2  ], 
                                    [0, 0,-1, 0.45  ],
                                    [0, 0, 0, 1    ]))
    
    else:
        pos_above_pickup = np.array(([1, 0, 0, 0.52 ],
                                        [0,-1, 0, 0.2  ], 
                                        [0, 0,-1, 0.52 ],
                                        [0, 0, 0, 1    ]))
        
        pos_above_drop = np.array(([1, 0, 0, 0.52 ],
                                    [0,-1, 0,-0.2  ], 
                                    [0, 0,-1, 0.45  ],
                                    [0, 0, 0, 1    ]))

    q_above_pickup = calculate_q_via_ik(pos_above_pickup, q_current)

    q_above_drop = calculate_q_via_ik(pos_above_drop, q_current)

    return q_above_pickup, q_above_drop


def move_to_place(T, q_current):
        if team == 'red':
            place_location = np.array(([1,0,0, 0.562],
                        [0,-1,0, 0.2], 
                        [0,0,-1,0.23 + T*0.053],
                        [0,0,0,1]))
        else:
            place_location = np.array(([1,0,0, 0.562],
                        [0,-1,0, -0.2], 
                        [0,0,-1,0.23 + T*0.053],
                        [0,0,0,1]))

        q_place = calculate_q_via_ik(place_location, q_current)

        return q_place

def set_dynamic_block_view(q_current):

    if team == 'red':
        q_above_rotate = np.array(([1, 0, 0, 0   ],
                                    [0,-1, 0, 0.7 ],
                                    [0, 0,-1, 0.4 ],
                                    [0, 0, 0, 1   ]))
        
        pos_above_drop_stacked = np.array(([1, 0, 0, 0.52 ],
                                    [0,-1, 0, 0.2  ], 
                                    [0, 0,-1, 0.65  ],
                                    [0, 0, 0, 1    ]))
    
    else:
        q_above_rotate = np.array(([1, 0, 0, 0   ],
                                    [0,-1, 0, -0.7],
                                    [0, 0,-1, 0.4 ],
                                    [0, 0, 0, 1   ]))
        
        pos_above_drop_stacked = np.array(([1, 0, 0, 0.52 ],
                                    [0,-1, 0,-0.2  ], 
                                    [0, 0,-1, 0.65  ],
                                    [0, 0, 0, 1    ]))
    
    q_above_rotate = calculate_q_via_ik(q_above_rotate, q_current)
    if q_above_rotate[-1] - pi < 2.897 and q_above_rotate[-1] - pi > -2.897:
        q_above_rotate[-1] = q_above_rotate[-1] - pi
    else:
        q_above_rotate[-1] = q_above_rotate[-1] + pi

    q_above_drop_stacked = calculate_q_via_ik(pos_above_drop_stacked, q_current)

    return q_above_rotate, q_above_drop_stacked

def dynamic_adjustment(x,y, w_t):


    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if r<0.22:
        return None, None
    # Update the angle
    theta += w_t
    
    # Convert back to Cartesian
    return r * np.cos(theta), r * np.sin(theta), theta

def move_to_dynamic_block(block, q_current):

    ee_rot = np.array(([1,0,0],
        			[0,-1,0], 
        			[0,0,-1],
                    [0, 0, 0]))
    block_pos = block[:,3]
    block_pos = block_pos.reshape(4,1)
    ee_goal = np.hstack((ee_rot,block_pos))

    print("ee_goal_before adjustment: ", ee_goal)
    x = ee_goal[0, 3]
    y = ee_goal[1, 3]
    y -= 0.990
    
    angle = rotation_matrix_to_angle_axis(ee_goal[:3,:3])

    xn,yn, theta =  dynamic_adjustment(x,y, 0.45)
    if xn == None:
        return None
    yn += 0.990

    ee_goal[0, 3] = xn
    ee_goal[1, 3] = yn
    print("ee_goal_ adjustment: ", ee_goal)
    
    ee_goal[2,3] = 0.22                             
    # angle = rotation_matrix_to_angle_axis(block[:3,:3])       
    angle = pi

    q_block = calculate_q_via_ik(ee_goal, q_current)
    if q_block is not None:
        q_block[-1] = q_block[-1] - angle 

    return q_block, theta, angle



def pick_place_dynamic(q_above_rotate, q_above_drop_stacked, q_above_drop, iterations = 1):
    print("Getting the block world position")

    arm.safe_move_to_position(q_above_rotate)

    block_count, block_world = get_block_world(q_above_rotate)

    block_detected_at = time_in_seconds()
    dynamic_start_time = time_in_seconds()

    itera = 0
    while itera < iterations:
        itera +=1
        if time_in_seconds() - dynamic_start_time > 150:
            break
        else:
            print("TIME PASSED: ", time_in_seconds() - dynamic_start_time)
        # If no blocks are detected for 20 seconds, break
        print("Time passed: ", time_in_seconds() - block_detected_at)
        if block_count == 0 and time_in_seconds() - block_detected_at > 20:
            break
        
        ####################################################################################################
        elif block_count == 0:
            continue
        # Pick Sequence
        print("Block Detected: ", block_count)
        block_detected_at = time_in_seconds()

        print("Moving to the block")
        q_block, theta, angle = move_to_dynamic_block(block_world[0], q_above_rotate)
        if np.all(q_block == None):
            continue
        arm.safe_move_to_position(q_block)
        
        q_block[-1] = q_block[-1] + theta + angle
        
        arm.safe_move_to_position(q_block)
        print("Closing the gripper")
        grab_block(grab_ee_dist, grab_ee_force)

        ####################################################################################################

        # Place Sequence

        # Move to the above drop stacked position
        print("Moving to the above drop stacked position")
        arm.safe_move_to_position(q_above_drop_stacked)

        # Detect where to place from the block world
        print("Detecting where to place")
        target_block_count, target_block_world = get_block_world(q_above_drop_stacked)

        if target_block_count == 0:
            q_place = move_to_place(0, q_above_drop)
        else:
            z_value =  int((max([block[2,3] - red_black_box_height for block in target_block_world]) * scaling_factor) + 1)            
            print("z_value: ", z_value)
            q_place = move_to_place(z_value, q_above_drop)

        # Move to the place location
        print("Moving to the place location")
        arm.safe_move_to_position(q_place)

        # Drop the block
        print("Dropping the block")
        # drop_block(drop_ee_dist, drop_ee_force)
        arm.open_gripper()

        ####################################################################################################

        # Reset sequence
        print("Resetting the sequence")
        arm.safe_move_to_position(q_above_drop_stacked)
        arm.safe_move_to_position(q_above_rotate)

        block_count, block_world = get_block_world(q_above_rotate)












if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE
    start_time = time_in_seconds()

    # variables
    team = team
    ik = IK()
    fk = FK()
    q_above_pickup = None
    q_above_drop = None
    q_above_rotate = None

    red_black_box_height = 0.2
    scaling_factor = 20

    drop_ee_dist = 0.09
    drop_ee_force = 10
    
    grab_ee_dist = 0.048
    grab_ee_force = 52

    q_above_rotate, q_above_drop_stacked = set_dynamic_block_view(start_position)
    q_above_pickup, q_above_drop = set_static_view(start_position)

    # ####################################################################################################

    # # # Static Pick and Place
    

    # # Move to the above pickup position
    # print("Moving to above pickup position")
    # arm.safe_move_to_position(q_above_pickup)

    # # Get the block world position
    # print("Getting the block world position")
    # block_count, block_world = get_block_world(q_above_pickup)

    # #EDIT THE BLOCK WORLD TO RETURN DICTIONARIES, WILL BE LOT MROE USEFUL WITH NOISE
    # ########################################################################

    # org_block_count = block_count

    # # Open the gripper
    # print("Opening the gripper")
    # drop_block(drop_ee_dist, drop_ee_force)

    # iteration = 0
    
    # while block_count > 0:
        
    #     ####################################################################################################

    #     # Pick Sequence
    #     print("Starting the pick sequence")

    #     # Move to the block
    #     print("Moving to the block")
    #     q_align, q_block = move_to_static_block(block_world[0], q_above_pickup)

    #     arm.safe_move_to_position(q_align)
    #     arm.safe_move_to_position(q_block)

    #     # Close the gripper
    #     print("Closing the gripper")
    #     grab_block(grab_ee_dist, grab_ee_force)

    #     #################################################################################################### 

    #     # Place Sequence
    #     print("Starting the place sequence")


    #     # Move to the above drop position
    #     print("Moving to above drop position")
    #     arm.safe_move_to_position(q_above_drop)

    #     # Detect where to place from the block world
    #     print("Detecting where to place")
    #     target_block_count, target_block_world = get_block_world(q_above_drop)


    #     print(target_block_count, target_block_world)

    #     if target_block_count == 0:
    #         q_place = move_to_place(0, q_above_drop)
    #     else:
    #         z_value =  int((max([block[2,3] - red_black_box_height for block in target_block_world]) * scaling_factor) + 1)            
    #         print("z_value: ", z_value)
    #         q_place = move_to_place(z_value, q_above_drop)

    #     # # Detect where to place from the iteration
    #     # q_place = move_to_place(iteration, q_above_drop)

    #     # Move to the place location
    #     print("Moving to the place location")
    #     arm.safe_move_to_position(q_place)

    #     # Drop the block
    #     print("Dropping the block")
    #     drop_block(drop_ee_dist, drop_ee_force)

    #     ####################################################################################################
        
    #     # Reset sequence 
    #     arm.safe_move_to_position(q_above_drop)
    #     arm.safe_move_to_position(q_above_pickup)

    #     block_count, block_world = get_block_world(q_above_pickup)

    #     iteration += 1
    

    ####################################################################################################

    # Dynamic Pick and Place
    

    # Get the block world position
    
    
    

    pick_place_dynamic(q_above_rotate, q_above_drop_stacked, q_above_drop, iterations = 2)
    # Move to the above drop stacked position
    arm.safe_move_to_position(q_above_drop_stacked)




    # get the transform from camera to panda_end_effector
    # H_ee_camera = detector.get_H_ee_camera()

    # Detect some blocks...
    # for (name, pose) in detector.get_detections():
    #      print(name,'\n',pose)

    # Uncomment to get middle camera depth/rgb images
    # mid_depth = detector.get_mid_depth()
    # mid_rgb = detector.get_mid_rgb()

    # Move around...
    end_time = time_in_seconds()
    print("Time taken: ", end_time - start_time, " seconds")
    # END STUDENT CODE