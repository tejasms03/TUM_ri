import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.spaces import Box
from gymnasium.utils import seeding
import os
import collections
import math
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class gym_hexapod_zoo(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
    	        
    	 
        self.physics_client = p.connect(p.GUI)
        #self.physics_client = p.connect(p.DIRECT) 

        self.observation_space_num = 52

        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float64)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_num,), dtype=np.float64)

        self.episodeNum = 0

        self.timeStep = 1/240
        p.setTimeStep(self.timeStep)

        self.L = 0.05
        
        self.LF_rot_id = 0
        self.LM_rot_id = 3 
        self.LB_rot_id = 6
        self.RF_rot_id = 9
        self.RM_rot_id = 12
        self.RB_rot_id = 15

        self.LF_lin_id = 1
        self.LM_lin_id = 4
        self.LB_lin_id = 7
        self.RF_lin_id = 10 
        self.RM_lin_id = 13
        self.RB_lin_id = 16

        self._env_step_counter=0

        self.hexapodId =0
        #p.setRealTimeSimulation(enableRealTimeSimulation=1)
       

        self.success_pos_error = 0.1
        self.steps_per_episode = 30
        self.num_interval_path_planning = 100
        self.time_flag = 0
        self.time_ecc = 0
        self.pos_error_norm_ecc = 0
        self.pre_pos_error_norm_ecc = 0
        self.dt_initial_ecc = 2*np.pi/110
        self.dt_cal_ecc = 0.02
        self.max_operator = 2*np.pi + np.pi/16
        self.time = np.linspace(0,self.max_operator, self.num_interval_path_planning)
        self.x_error_sum = np.zeros(10)
        self.y_error_sum = np.zeros(10)
        self.interval_count_each_exp = np.zeros(10)
        self.sum_reward_each_exp = np.zeros(10)
        self.sum_step_each_exp = np.zeros(10)
        self.sum_reward_each_episode = 0
        self.sum_step_each_episode = 0
        self.sum_pos_error_mag_per_exp = np.zeros(10)
        self.pre_pos_target =  np.zeros(2)

        # plot init parameters
        self.trajectory = []
        self.robot_path = [[] for _ in range(10)]
        self.trajectory = []
        self.robot_path = [[]for _ in range(10)]
        self.x_ref = []
        self.x_robot = [[]for _ in range(10)]
        self.y_ref = []
        self.y_robot = [[]for _ in range(10)]
        self.x_error_plot = [[]for _ in range(10)]
        self.y_error_plot = [[]for _ in range(10)]
        self.reward_each_time_interval = [[]for _ in range(10)]
        self.interval_control_ecc = [[]for _ in range(10)]
        self.pos_error_mag_per_exp = [[]for _ in range(10)]
        self.num_exp = 1
        self.color = ['b','g','r','c','m','y','k']

        
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.countdown = 0
        #print(self.sum_reward_each_episode,self.sum_step_each_episode)

        self.num_interval_path_planning= [0,30,200,'-ATP','ECC_1','ECC_10']

        print("num_exp =",self.num_exp)
        if self.num_exp == 1:
            self.time = np.linspace(0, self.max_operator, self.num_interval_path_planning[self.num_exp])
            self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)
        print("num_exp =",self.num_exp)
        if self.num_exp == 2:
            self.time = np.linspace(0, self.max_operator, self.num_interval_path_planning[self.num_exp])
            self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)
        # if self.num_exp == 3:
        #     self.time = np.linspace(0, self.max_operator, self.num_interval_path_planning[self.num_exp])
        #     self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)
        # if self.num_exp == 4:
        #     self.time = np.linspace(0, self.max_operator, self.num_interval_path_planning[self.num_exp])
        #     self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)
        # if self.num_exp == 5:
        #     self.time = np.linspace(0, self.max_operator, self.num_interval_path_planning[self.num_exp])
        #     self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)
        # print("num_exp =",self.num_exp,"tim_ecc=",self.time_ecc)
        if self.num_exp == 3:
            self.time_ecc, self.dt_cal_ecc, self.pos_init,self.pos_target, self.pre_pos_error_norm_ecc = self._set_init_state_with_ECC(
                self.pos_error_norm_ecc,self.pre_pos_error_norm_ecc, self.time_ecc,self.dt_cal_ecc,self.dt_initial_ecc)
        
        # #print("num_exp =",self.num_exp)
        # if self.num_exp == 2:
        #     self.time_ecc, self.dt_cal_ecc, self.pos_init,self.pos_target, self.pre_pos_error_norm_ecc = self._set_init_state_with_ECC(
        #         self.pos_error_norm_ecc,self.pre_pos_error_norm_ecc, self.time_ecc,self.dt_cal_ecc,self.dt_initial_ecc)
        # #print("num_exp =",self.num_exp)
        # if self.num_exp == 3:
        #     self.time_ecc, self.dt_cal_ecc, self.pos_init,self.pos_target, self.pre_pos_error_norm_ecc = self._set_init_state_with_ECC(
        #         self.pos_error_norm_ecc,self.pre_pos_error_norm_ecc, self.time_ecc,self.dt_cal_ecc,self.dt_initial_ecc)
        
        #if self.time_ecc > self.max_operator:
        #print("num_exp =",self.num_exp)
        if self.num_exp == 4:  
            for i in range (1,4):
                print("x_error_sum path",i," = ",np.round(self.x_error_sum[i],4),
                  "|| y_error_sum path",i," = ",np.round(self.y_error_sum[i],4),
                  "|| inteval count path",i," = ",np.round(self.interval_count_each_exp[i],4),
                  "|| x_error average path",i," = ", np.round(self.x_error_sum[i]/self.interval_count_each_exp[i],4),
                  "|| y_error average path",i," = ", np.round(self.y_error_sum[i]/self.interval_count_each_exp[i],4),
                  "|| error magnitude average path",i," = ", np.round(self.sum_pos_error_mag_per_exp[i]/self.interval_count_each_exp[i],4),
                  "|| average reward path",i," = ", np.round(self.sum_reward_each_exp[i]/self.sum_step_each_exp[i],4),
                  "|| sum of step",i," = ", np.round(self.sum_step_each_exp[i],4),
                )
            self._plot_path()
    
                
        # self.time_flag, self.pos_init, self.pos_target = self._set_init_state_without_ECC(self.time,self.time_flag)

        self.action_pre = np.zeros(12)
        self.joint_pos_pre = np.zeros(12)
        
        observation = np.zeros(self.observation_space_num)
        self.episodeNum_count=0
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        self.reward =0
        
        info = {}
     
        return observation, info

    def step(self, action):
        
        dv_rot = math.radians(15)#*self.timeStep
        dv_lin = 0.01#*self.timeStep 
    
        LF_rot_action = action[0] * dv_rot 
        LM_rot_action = action[1] * dv_rot 
        LB_rot_action = action[2] * dv_rot 
        RF_rot_action = action[3] * dv_rot 
        RM_rot_action = action[4] * dv_rot 
        RB_rot_action = action[5] * dv_rot 

        calib=0.05-dv_lin/2

        LF_lin_action = -(dv_lin*action[6]/2 + calib) 
        LM_lin_action = -(dv_lin*action[7]/2 + calib)
        LB_lin_action = -(dv_lin*action[8]/2 + calib)
        RF_lin_action = -(dv_lin*action[9]/2 + calib)
        RM_lin_action = -(dv_lin*action[10]/2 + calib)
        RB_lin_action = -(dv_lin*action[11]/2 + calib)
        
        LF_rot_motor_pos = LF_rot_action
        LM_rot_motor_pos = LM_rot_action
        LB_rot_motor_pos = LB_rot_action
        RF_rot_motor_pos = RF_rot_action
        RM_rot_motor_pos = RM_rot_action
        RB_rot_motor_pos = RB_rot_action
        LF_lin_motor_pos = LF_lin_action
        LM_lin_motor_pos = LM_lin_action
        LB_lin_motor_pos = LB_lin_action
        RF_lin_motor_pos = RF_lin_action
        RM_lin_motor_pos = RM_lin_action
        RB_lin_motor_pos = RB_lin_action

        # pos_set = np.array([
        #     LF_rot_motor_pos,
        #     LM_rot_motor_pos,
        #     LB_rot_motor_pos,
        #     RF_rot_motor_pos,
        #     RM_rot_motor_pos,
        #     RB_rot_motor_pos,
        #     LF_lin_motor_pos,
        #     LM_lin_motor_pos,
        #     LB_lin_motor_pos,
        #     RF_lin_motor_pos,
        #     RM_lin_motor_pos,
        #     RB_lin_motor_pos,
        #     ])

        #determine the action_time

        vel_max_rot = 5
        vel_max_lin = 0.06

        simulation_step_period = 1/240 # defaul value from pybullet, look up the quick tutorials
        
       
        # Record the last position and orientation
        self.state_body = p.getBasePositionAndOrientation(self.hexapodId)        
        body_position = self.state_body[0]
        body_orientation = p.getEulerFromQuaternion(self.state_body[1])

        last_x = body_position[0]
        last_y = body_position[1]
        last_orientation = body_orientation[2]

        self.joint_pos_pre = self._get_joint_pos() 
        
        simulation_step_num = 240
        

        for i  in range(simulation_step_num):
            #set command for actuators
            p.setJointMotorControl2(self.hexapodId, self.LF_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=LF_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.LM_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=LM_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.LB_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=LB_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.RF_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=RF_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.RM_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=RM_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.RB_rot_id, controlMode=p.POSITION_CONTROL, targetPosition=RB_rot_motor_pos, maxVelocity=vel_max_rot)
            p.setJointMotorControl2(self.hexapodId, self.LF_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=LF_lin_motor_pos, maxVelocity=vel_max_lin)
            p.setJointMotorControl2(self.hexapodId, self.LM_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=LM_lin_motor_pos, maxVelocity=vel_max_lin)
            p.setJointMotorControl2(self.hexapodId, self.LB_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=LB_lin_motor_pos, maxVelocity=vel_max_lin)
            p.setJointMotorControl2(self.hexapodId, self.RF_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=RF_lin_motor_pos, maxVelocity=vel_max_lin)
            p.setJointMotorControl2(self.hexapodId, self.RM_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=RM_lin_motor_pos, maxVelocity=vel_max_lin)
            p.setJointMotorControl2(self.hexapodId, self.RB_lin_id, controlMode=p.POSITION_CONTROL, targetPosition=RB_lin_motor_pos, maxVelocity=vel_max_lin)
            #run a simulation step
            #time.sleep(0.01)
            p.stepSimulation()
  

        action_time = simulation_step_num * simulation_step_period

        # Calculate the reward
        self.state_body = p.getBasePositionAndOrientation(self.hexapodId)        
        body_position = self.state_body[0]
        
        current_x = body_position[0]
        current_y = body_position[1]

        pos_current = np.array([current_x,current_y])
        
        joint_pos = self._get_joint_pos()

        joint_vel = (joint_pos - self.joint_pos_pre)/action_time

        #print("pos=",np.round(joint_pos-pos_set,3),self.countdown,action_time)
        #print(" pos set=",np.round(pos_set,3),self.countdown)

        #penalty for sideways rotation of the body 
        body_orientation = p.getEulerFromQuaternion(self.state_body[1])
        get_axis_angle = p.getAxisAngleFromQuaternion(self.state_body[1])
        current_orientation = body_orientation[2]
        
        
        #Determine the linear velocity and angular velocity of robot base after conducted a new action
        # THis is last update of calculation on 20240422 
        body_linear_velocity = np.array([(current_x-last_x)/action_time,(current_y-last_y)/action_time])
        body_angular_velocity = (current_orientation -  last_orientation)/action_time
         
        rot_matrix = p.getMatrixFromQuaternion(self.state_body[1])
        local_up_vec = rot_matrix[6:]
        self.shake_val= np.dot(np.asarray([0, 0, 1]), np.asarray(local_up_vec))
        
        pos_error = self.pos_target - pos_current
        pos_error_norm = np.linalg.norm(self.pos_target - pos_current)
        self.pos_error_norm_ecc = pos_error_norm
        #---REWARD-----
       
        reward, info = self._get_rw_norm(action,pos_error_norm,pos_error,body_linear_velocity,self.pos_init,self.pos_target,self.countdown)
        
         
        #----OBSERVATION------ 
        
        observation = self._get_obs(self.action_pre,pos_error,pos_error_norm, body_linear_velocity,joint_vel,\
                                    self.countdown,self.pos_init,self.pos_target)      

        # if self._env_step_counter % 100 == 0:
        #     print("reward=",reward,"body lin vel =",np.round(body_linear_velocity,6),"pos error=",\
        #         np.round(np.linalg.norm(pos_error),6),"pos_target=",np.round(self.pos_target,6))
        #     print("target_reward",np.round(target_reward,6),"penalty_reward=",np.round(penalty_reward,6),\
        #         "stall_reward=",np.round(stall_reward,6),"reach_reward=",np.round(reach_reward,6),\
        #         "orientation_reward=",round(orientation_reward,6))
        #     print("step_counter = ",self._env_step_counter,"  linear error = ",round(vel_base_error_track,6), " rot error =",round(rot_base_error_track,6), "reward = ",round(reward,5))
        #     print("lin vel get",round(self.get_base_vel,6),"rot vel get=",round(self.get_base_rot,6))
        # # print("action = ",action,"  reward= ",reward,"  linear vel = ",get_base_vel,"   v_r", v_l,"  v_l",v_r)
        #print("velocity tracking= ",self.v_l_ref-v_l, " ",self.v_r_ref-v_r)

        # DEINFE TEMINATED OR TRUNCATED
        terminated , truncated = self._define_terminated_truncated_random_training(self.countdown,self.steps_per_episode, pos_error_norm, self.success_pos_error)
        #terminated , truncated, self.time_flag, self.countdown = self._define_terminated_truncated_8_shape(self.time_flag,self.countdown)
        if self._termination():
            terminated =True
            truncated = False
            #reward =  - 10*self.steps_per_episode
            #reward =  - 0*10*self.steps_per_episode

            reset_start_pos= np.array([self._get_base_pos_x_y()[0],self._get_base_pos_x_y()[1], 0.1])
            reset_start_orientation = p.getQuaternionFromEuler(np.array([0,0,self._get_base_orientation_z()]))
            p.resetSimulation()
            self._load_robot(reset_start_pos, reset_start_orientation)
        
        self.action_pre = action
        self.countdown += 1
        self.sum_reward_each_episode += reward
        self.sum_step_each_episode += 1
        self.joint_pos_pre = joint_pos 
        #print("step_num",self.countdown)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return observation, reward, terminated, truncated, info
     
    def _get_rw(self,action,pos_error_norm,pos_error,body_linear_velocity):
       

        self.target_pos_error =0.1

        #Target reward
        #if pos_error_norm < self.target_pos_error:
        target_reward = np.exp(-20*pos_error_norm**2)
        #target_reward = -pos_error_norm
        #else:
            #target_reward = 0

        # penalty action
        penalty_reward = -np.square(action-self.action_pre).sum()

        # orientation reward
        cos_alpha = np.dot(body_linear_velocity,pos_error)/(np.linalg.norm(body_linear_velocity)*np.linalg.norm(pos_error))
        orientation_reward = cos_alpha
        

        # Stalling penalty
        if np.linalg.norm(body_linear_velocity) < 0.005 and pos_error_norm > self.success_pos_error:
            stall_reward = -1
        else:
            stall_reward = 0
        

        if pos_error_norm < self.success_pos_error:
            terminated = True
            reach_reward=1
            print("reach target")
        else:
            reach_reward =0

        reward = 0.85*target_reward + 0*penalty_reward + 0.05*stall_reward + reach_reward + 0.1*orientation_reward 

        return reward
    
    def _get_rw_norm(self,action,pos_error_norm,pos_error,body_linear_velocity,pos_init,pos_target,countdown):
       
        robot_vec= self._cal_robot_coordinate_world_coordinate()
        self.target_pos_error = 0.1
        mag_pos_init_goal = self._get_mag_pos_init_goal(pos_init,pos_target)

        #Target reward
        target_reward = 1 - pos_error_norm/mag_pos_init_goal #0.8 is the max error to get positive reward

        # penalty action
        penalty_reward = -np.square(action-self.action_pre).sum()

        # orientation reward
        cos_alpha = np.dot(body_linear_velocity,pos_error)/(np.linalg.norm(body_linear_velocity)*np.linalg.norm(pos_error))
        orientation_reward = cos_alpha
        
        self.success_pos_error= 0.01

        # reward for push robot to move forward
        

        cos_beta = np.dot(robot_vec,pos_error)/(np.linalg.norm(robot_vec)*np.linalg.norm(pos_error))
        move_forward_reward = cos_beta

        # Stalling penalty
        if np.linalg.norm(body_linear_velocity) < 0.0005 and pos_error_norm > self.success_pos_error:
            stall_reward = -1
        else:
            stall_reward = 0
        

        if pos_error_norm < self.success_pos_error:
            terminated = True
            reach_reward=1
            print("reach target")
        else:
            reach_reward =0
            terminated = False

        # run_time_pennatly

        run_time_reward = - countdown/self.steps_per_episode

        reward = 0.8*target_reward \
            + 0*penalty_reward \
                + .025*stall_reward \
                    + reach_reward \
                        + 0.75*orientation_reward \
                            + 0.75*move_forward_reward\
                                + 0.25*run_time_reward
        
        info ={
            "target_reward": target_reward,
            "penalty_reward": penalty_reward,
            "stall_reward": stall_reward,
            "reach_reward": reach_reward,
            "orientation_reward": orientation_reward,
            "move_forward_reward": move_forward_reward,
            "run_time_reward": run_time_reward,
           #"pos_target": pos_target,
        }
        #print(info,robot_vec)
        return reward, info
    
    def _get_obs(self,action_pre,pos_error,pos_error_norm, body_linear_velocity,joint_vel,countdown,pos_init,pos_target):

        joint_pos = self._get_joint_pos()
        base_pos = self._get_base_pos_x_y()
        base_ori = self._get_base_orientation_z()
        pos_current = self._get_base_pos_x_y()
        distance_pos_init_target = self._get_mag_pos_init_goal(pos_init,pos_target)

        data_state = np.array([action_pre]) #12
        data_state = np.append(data_state, joint_pos) # 12
        data_state = np.append(data_state, base_pos) # 2
        data_state = np.append(data_state, base_ori) # 1
        data_state = np.append(data_state, joint_vel) # 12
        data_state = np.append(data_state, pos_init) #2
        data_state = np.append(data_state, pos_target) #2
        data_state = np.append(data_state, pos_current) #2
        data_state = np.append(data_state, pos_error) #2
        data_state = np.append(data_state, pos_error_norm) #1
        data_state = np.append(data_state, distance_pos_init_target)#1
        data_state = np.append(data_state, body_linear_velocity) #2
        data_state = np.append(data_state, countdown) #1
        # print("pos_init",np.round(pos_init,3),
        #       "pos_target",np.round(pos_target,3),
        #       "pos_error",np.round(pos_error_norm,3),
        #       "pos_current", np.round(pos_current,6),
        #       )
        return np.ravel(data_state)

    def _get_joint_pos(self):

        LF_rot_pos = p.getJointState(self.hexapodId,self.LF_rot_id)[0] 
        LM_rot_pos = p.getJointState(self.hexapodId,self.LM_rot_id)[0]
        LB_rot_pos = p.getJointState(self.hexapodId,self.LB_rot_id)[0] 
        RF_rot_pos = p.getJointState(self.hexapodId,self.RF_rot_id)[0]
        RM_rot_pos = p.getJointState(self.hexapodId,self.RM_rot_id)[0]
        RB_rot_pos = p.getJointState(self.hexapodId,self.RB_rot_id)[0] 
        LF_lin_pos = p.getJointState(self.hexapodId,self.LF_lin_id)[0] 
        LM_lin_pos = p.getJointState(self.hexapodId,self.LM_lin_id)[0] 
        LB_lin_pos = p.getJointState(self.hexapodId,self.LB_lin_id)[0]
        RF_lin_pos = p.getJointState(self.hexapodId,self.RF_lin_id)[0]
        RM_lin_pos = p.getJointState(self.hexapodId,self.RM_lin_id)[0] 
        RB_lin_pos = p.getJointState(self.hexapodId,self.RB_lin_id)[0]

        joint_pos = np.array(
            [
                LF_rot_pos, LM_rot_pos,LB_rot_pos,RF_rot_pos,RM_rot_pos,RB_rot_pos,
                LF_lin_pos, LM_lin_pos,LB_lin_pos,RF_lin_pos,RM_lin_pos,RB_lin_pos,
            ]
            )

        return joint_pos
    
    def _get_joint_vel(self):

        LF_rot_vel = p.getJointState(self.hexapodId,self.LF_rot_id)[0] 
        LM_rot_vel = p.getJointState(self.hexapodId,self.LM_rot_id)[0]
        LB_rot_vel = p.getJointState(self.hexapodId,self.LB_rot_id)[0] 
        RF_rot_vel = p.getJointState(self.hexapodId,self.RF_rot_id)[0]
        RM_rot_vel = p.getJointState(self.hexapodId,self.RM_rot_id)[0]
        RB_rot_vel = p.getJointState(self.hexapodId,self.RB_rot_id)[0] 
        LF_lin_vel = p.getJointState(self.hexapodId,self.LF_lin_id)[0] 
        LM_lin_vel = p.getJointState(self.hexapodId,self.LM_lin_id)[0] 
        LB_lin_vel = p.getJointState(self.hexapodId,self.LB_lin_id)[0]
        RF_lin_vel = p.getJointState(self.hexapodId,self.RF_lin_id)[0]
        RM_lin_vel = p.getJointState(self.hexapodId,self.RM_lin_id)[0] 
        RB_lin_vel = p.getJointState(self.hexapodId,self.RB_lin_id)[0]

        joint_vel = np.array(
            [
                LF_rot_vel, LM_rot_vel,LB_rot_vel,RF_rot_vel,RM_rot_vel,RB_rot_vel,
                LF_lin_vel, LM_lin_vel,LB_lin_vel,RF_lin_vel,RM_lin_vel,RB_lin_vel,
            ]
        )

        return joint_vel
    
    def _get_base_pos_x_y(self):

        base_pos = p.getBasePositionAndOrientation(self.hexapodId)[0]

        return np.array([base_pos[0], base_pos[1]])
    
    def _get_base_orientation_z(self):
        
        base_ori = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.hexapodId)[1])

        return base_ori[2]

    def close(self):
        p.disconnect()
    
    def is_fallen(self):
        """Decide whether hexapod has fallen.
        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), hexapod is considered fallen.
        Returns:
          Boolean value that indicates whether rex has fallen.
        """
        return self.shake_val < 0.85 
             
    def _termination(self):
        if self.is_fallen():
            print("FALLING DOWN!")
        return self.is_fallen()   

    def _generate_eight_shape_path(self,time):
        # Define the scale factors for the x and y coordinates
        scale_x = 2  # Scale factor for x-coordinate
        scale_y = 2.5 # Scale factor for y-coordinate

        # Define x and y coordinates for the 8-shaped path with scaled dimensions
        x = scale_x * np.sin(time)
        y = scale_y * np.sin(time) * np.cos(time)
        return x, y
    def _generate_line_shape_path(self,time):
        # Define the scale factors for the x and y coordinates
        scale_x = 2  # Scale factor for x-coordinate
        scale_y = 2.5 # Scale factor for y-coordinate

        # Define x and y coordinates for the 8-shaped path with scaled dimensions
        x = time
        y = 0
        return x, y
    def _generate_sin_shape_path(self,time):
        # Define the scale factors for the x and y coordinates
        scale_x = 2  # Scale factor for x-coordinate
        scale_y = 2.5 # Scale factor for y-coordinate

        # Define x and y coordinates for the 8-shaped path with scaled dimensions
        x = time*2/np.pi
        y = np.sin(time)
        return x, y
    def _generate_zig_shape_path(self,time):
        # Define the scale factors for the x and y coordinates
        scale_x = 2  # Scale factor for x-coordinate
        scale_y = 2.5 # Scale factor for y-coordinate
        x = time
        if int(x)%2==0:
            m=1
            c=-1*int(x)
        else:
            m=-1
            c=int(x)+1
        # Define x and y coordinates for the 8-shaped path with scaled dimensions
        y = m*x+c
        return x, y
    
    def _generate_target_training(self):
         
        pos_init = np.array([random.uniform(-1,1),random.uniform(-1,1)])
        pos_target = np.zeros(2)
        # while np.linalg.norm(pos_target - pos_init) < 0.15 \
        #     and np.linalg.norm(pos_target - pos_init) > 1:
        pos_target= np.array([random.uniform(-1,1),random.uniform(-1,1)])
        
        return pos_init, pos_target
    
    def _get_mag_pos_init_goal(self,pos_init,pos_target):
        
        return np.linalg.norm(pos_init-pos_target)

    def _cal_robot_coordinate_world_coordinate(self):
        
        pos_current = self._get_base_pos_x_y()
        alpha= self._get_base_orientation_z()
        x_r_norm_vec = [1,0]
        x_r_tip_world_frame = np.array([ 
            np.cos(alpha)*x_r_norm_vec[0] - np.sin(alpha)*x_r_norm_vec[1],
            np.sin(alpha)*x_r_norm_vec[0] + np.cos(alpha)*x_r_norm_vec[1], 
            ])  

        return  x_r_tip_world_frame 

    def _define_terminated_truncated_8_shape(self, time_flag,countdown):
        
        terminated = False
        truncated = False
        if countdown > self.steps_per_episode:
            time_flag+=1
            countdown = 0
            # print("reward=",reward,"body lin vel =",np.round(body_linear_velocity,6),"pos error=",\
            #       np.round(np.linalg.norm(pos_error),6),"pos_target=",np.round(self.pos_target,6))
            # print(self.time_flag)

        if time_flag > 200:
            truncated = True
            terminated = False
            time_flag=0
            print("full episode without reaching target:truncated")

        else:
            truncated = False
        
        return terminated, truncated, time_flag, countdown

    def _define_terminated_truncated_random_training(self, countdown, steps_per_episode,pos_error_norm,success_pos_error):

        terminated = False
        truncated = False

        if countdown > steps_per_episode:
            truncated = True

        if pos_error_norm < success_pos_error:
            terminated = True

        return terminated, truncated

    def _load_robot(self,startPos,startOrientation):

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0.,0.,-9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        

        hexapodPath= "/home/nvidia/rl-baselines3-zoo/rl_zoo3/URDF/urdf/Mobile_hexapod_robot_v3_20240112.urdf"
        self.hexapodId = p.loadURDF(hexapodPath.format(6, 6),startPos, startOrientation,useFixedBase=False)
        #p.setPhysicsEngineParameter(enableConeFriction=0)
        
        rest_poses_rot = 0
        rest_poses_lin = 0.04
        p.resetJointState(self.hexapodId,self.LF_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.LM_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.LB_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.RF_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.RM_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.RB_rot_id, rest_poses_rot)
        p.resetJointState(self.hexapodId,self.LF_lin_id, rest_poses_lin)
        p.resetJointState(self.hexapodId,self.LM_lin_id, rest_poses_lin)
        p.resetJointState(self.hexapodId,self.LB_lin_id, rest_poses_lin)
        p.resetJointState(self.hexapodId,self.RF_lin_id, rest_poses_lin)
        p.resetJointState(self.hexapodId,self.RM_lin_id, rest_poses_lin)
        p.resetJointState(self.hexapodId,self.RB_lin_id, rest_poses_lin)

    def _set_pos_orientation_load_robot_training(self):

        body_z_init_rot= random.uniform(-math.pi,math.pi)
        startOrientation = p.getQuaternionFromEuler(np.array([0.,0.,body_z_init_rot]))
        pos_init, pos_target= self._generate_target_training()
        
        startPos = [pos_init[0],pos_init[1],0.1]
        
        return  startPos, startOrientation, pos_init, pos_target
    
    def _set_pos_orientation_load_robot_reset(self):

        startOrientation = p.getQuaternionFromEuler(np.array([0.,0.,0.]))
        startPos = np.array([0,0,0.1])
        p.resetSimulation()
        return startPos, startOrientation

    def _generate_target_with_ECC(self,t):
        A = 3
        B = 2
        a = 2
        b = 1
        delta = np.pi / 2
        x = A * np.sin(a * t)
        y = B * np.sin(b * t + delta)
        return x, y

    def _cal_kc_ecc(self,pos_error_norm):
        n_max = 200
        n_min = 30
        phi = self.max_operator
        error_max = 1

        return phi*(n_max - n_min)/((n_max*n_min)*error_max)

    def _error_compensation_controller(self,pos_error_norm,pre_pos_error_norm,dt_cal_ecc,dt_initial_ecc,time_ecc):
        k_p = 0#-0.8#-6.4321608040201e-05 #-0.001#-0.005 
        k_d = 0#-0.5#-3.2160804020100515e-05#-0.004#-0.0001
        k_i = 0#-0.0

        #pos_error_norm = max(0.0001,pos_error_norm)
        k_c = self._cal_kc_ecc(pos_error_norm)
        #error_derivative = (pos_error_norm - pre_pos_error_norm)/dt_cal_ecc 
        #self.integral_term += pos_error_norm*dt_cal_ecc
        # if 0 < time_ecc < 5*np.pi/6:
        #     dt_initial_ecc = self.max_operator/50
        # else:
        #     dt_initial_ecc = self.max_operator/180 
        # dt_initial_ecc = self.max_operator/30
        # dt_cal_ecc = min(max(self.max_operator/200, dt_initial_ecc + k_p * pos_error_norm + k_d* error_derivative
        #                      + k_i*self.integral_term),self.max_operator/30)  # Ensure dt doesn't become negative

        dt_cal_ecc = min(max(self.max_operator/200, self.max_operator/200 +  k_c*pos_error_norm),self.max_operator/30)
        time_ecc += dt_cal_ecc    

        return time_ecc, dt_cal_ecc
    
    def _set_init_state_without_ECC(self,time,time_flag):

        
        if time_flag == 0:
            startPos, startOrientation = self._set_pos_orientation_load_robot_reset()
            self._load_robot(startPos,startOrientation)
            self.pre_pos_target = np.zeros(2)

        pos_target = self._generate_sin_shape_path(time[time_flag])       
        pos_init = self._get_base_pos_x_y()
        
        #self.trajectory.append((pos_target[0], pos_target[1]))
        self.robot_path[self.num_exp].append((pos_init[0], pos_init[1]))
        
        #self.x_ref.append((time[time_flag],pos_target[0]))
        self.x_robot[self.num_exp].append((time[time_flag],pos_init[0]))
        
        #self.y_ref.append((time[time_flag],pos_target[1]))
        self.y_robot[self.num_exp].append((time[time_flag],pos_init[1]))

        self.pos_error_mag_per_exp[self.num_exp].append((time[time_flag],np.linalg.norm(self.pre_pos_target-pos_init)))
        
        self.x_error_plot[self.num_exp].append((time[time_flag],self.pre_pos_target[0]-pos_init[0]))
        
        self.y_error_plot[self.num_exp].append((time[time_flag],self.pre_pos_target[1]-pos_init[1]))

        self.reward_each_time_interval[self.num_exp].append((time[time_flag],self.sum_reward_each_episode))


        self.x_error_sum[self.num_exp] += self.pre_pos_target[0]-pos_init[0]
        self.y_error_sum[self.num_exp] += self.pre_pos_target[1]-pos_init[1]
        self.sum_pos_error_mag_per_exp[self.num_exp] += np.linalg.norm(self.pre_pos_target-pos_init)
        self.interval_count_each_exp[self.num_exp] += 1
        self.sum_reward_each_exp[self.num_exp] += self.sum_reward_each_episode
        self.sum_step_each_exp[self.num_exp] += self.sum_step_each_episode  
        
        self.sum_reward_each_episode = 0
        self.sum_step_each_episode = 0

        time_flag +=1
        self.pre_pos_target = pos_target

        if time_flag == self.num_interval_path_planning[self.num_exp]:
            #self._plot_path(self.trajectory,self.robot_path,self.num_exp)
            time_flag = 0
            self.num_exp +=1

        return time_flag, pos_init, pos_target

    def _set_init_state_with_ECC(self,pos_error_norm_ecc, pre_pos_error_norm_ecc,time_ecc,dt,dt_initial):
        t_final = self.max_operator
        
        if time_ecc == 0:
            startPos, startOrientation = self._set_pos_orientation_load_robot_reset()
            self._load_robot(startPos,startOrientation)
            self.pre_pos_target = np.zeros(2)
            self.integral_term=0
  
        time_ecc,dt_cal_ecc = self._error_compensation_controller(pos_error_norm_ecc,pre_pos_error_norm_ecc,dt,dt_initial,time_ecc)
        
        pos_target = self._generate_sin_shape_path(time_ecc)
        pos_init = self._get_base_pos_x_y()

        self.trajectory.append((pos_target[0], pos_target[1]))
        self.robot_path[self.num_exp].append((pos_init[0], pos_init[1]))
        
        self.x_ref.append((time_ecc,pos_target[0]))
        self.x_robot[self.num_exp].append((time_ecc,pos_init[0]))
        
        self.y_ref.append((time_ecc,pos_target[1]))
        self.y_robot[self.num_exp].append((time_ecc,pos_init[1]))

        self.pos_error_mag_per_exp[self.num_exp].append((time_ecc,np.linalg.norm(self.pre_pos_target-pos_init)))
        
        self.x_error_plot[self.num_exp].append((time_ecc,self.pre_pos_target[0]-pos_init[0]))
        
        self.y_error_plot[self.num_exp].append((time_ecc,self.pre_pos_target[1]-pos_init[1]))

        num_exp_without_ecc = 2
        if self.num_exp > num_exp_without_ecc:
            self.interval_control_ecc[self.num_exp].append((time_ecc,dt_cal_ecc))

        self.reward_each_time_interval[self.num_exp].append((time_ecc,self.sum_reward_each_episode))

        self.x_error_sum[self.num_exp] += self.pre_pos_target[0]-pos_init[0]
        self.y_error_sum[self.num_exp] += self.pre_pos_target[1]-pos_init[1]
        self.sum_pos_error_mag_per_exp[self.num_exp] += np.linalg.norm(self.pre_pos_target-pos_init)
        self.interval_count_each_exp[self.num_exp] += 1
        self.sum_reward_each_exp[self.num_exp] += self.sum_reward_each_episode
        self.sum_step_each_exp[self.num_exp] += self.sum_step_each_episode  

        #print(self.sum_reward_each_episode,self.sum_step_each_episode)


        pre_pos_error_norm_ecc = pos_error_norm_ecc
        #print(dt_cal_ecc,dt_cal_ecc)
        self.sum_reward_each_episode = 0
        self.sum_step_each_episode = 0

        self.pre_pos_target = pos_target

        if time_ecc > t_final:
            # self._plot_path(self.trajectory_ecc,self.robot_path_ecc,self.num_exp)
            # self._plot_path(self.x_ref_ecc,self.x_robot_ecc,self.num_exp)
            # self._plot_path(self.y_ref_ecc,self.y_robot_ecc,self.num_exp)
            # self._plot_path(self.x_ref_ecc-self.x_robot_ecc,self.time_ecc,self.num_exp)
            # self._plot_path(self.y_ref_ecc-self.y_robot_ecc,self.time_ecc,self.num_exp)
            time_ecc = 0
            self.num_exp +=1

        return time_ecc,dt_cal_ecc, pos_init, pos_target, pre_pos_error_norm_ecc
    


    def _plot_path(self):

        # Define linestyle for each element
        linestyle = [':','-.','-','-','-','-']
        marker = ['','','','','','']
        self.color = ['b','g','r','m','c','y','k']
        figure_size = (6,4)
        marker_size = 1

        # figure 1: 8-shaped path of reference and robot
        figure1 = plt.figure(figsize=figure_size)

        traj_x, traj_y = zip(*self.trajectory)
        plt.plot(traj_x, traj_y, label='Ref path', linestyle='--',color='b')

        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.robot_path[i])
            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)

            
            #plt.plot(robot_x_coords, robot_y_coords, label='Raw Robot Path', linestyle=':', alpha=0.7)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i],marker=marker[i-1], markersize = marker_size)
            #plt.title('Robot Following Eight-Shaped Path')
            self._add_title('Robot Following Eight-Shaped Path')
            plt.xlabel('x (meter)')
            plt.ylabel('y (meter)')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            plt.axis('equal')
        
        # Figure 2: plot the x trajectory of reference and robot
        figure2 = plt.figure(figsize=figure_size)
        traj_x, traj_y = zip(*self.x_ref)
        #traj_x =self.moving_average(traj_x)
        #traj_y =self.moving_average(traj_y)
        plt.plot(traj_x, traj_y, label='Ref x', linestyle='--', color = 'b')

        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.x_robot[i])
            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)

            
            #plt.plot(robot_x_coords, robot_y_coords, label='Raw Robot Path', linestyle=':', alpha=0.7)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title(' X-Trajectory Tracking of Reference and Robot')
            self._add_title('X-Trajectory Tracking of Refrence and Robot')
            plt.xlabel('time')
            plt.ylabel('x (meter)')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            plt.axis('equal')

        #Figure 3: Plot Y-trajectory of Reference and Robot
        figure3 = plt.figure(figsize=figure_size)
        traj_x, traj_y = zip(*self.y_ref)
        #traj_x =self.moving_average(traj_x)
        #traj_y =self.moving_average(traj_y)

        plt.plot(traj_x, traj_y, label='Ref y', linestyle='--', color ='b')
        
        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.y_robot[i])
            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)

            
            #plt.plot(robot_x_coords, robot_y_coords, label='Raw Robot Path', linestyle=':', alpha=0.7)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title('Y-Trajectory Tracking of Refrence and Robot')
            self._add_title('Y-Trajectory Tracking of Refrence and Robot')
            plt.xlabel('time')
            plt.ylabel('y (meter)')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            plt.axis('equal')

        # Figure 4: Plot X-Error-Tracking of Robot
        figure4 = plt.figure(figsize=figure_size)
                
        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.x_error_plot[i])
            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)

            
            #plt.plot(robot_x_coords, robot_y_coords, label='Raw Robot Path', linestyle=':', alpha=0.7)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title('X-Error-Tracking of Robot')
            self._add_title('X-Error-Tracking of Robot')
            plt.xlabel('time')
            plt.ylabel('x_error (meter)')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            #plt.axis('equal')

        #Figure 5: Plot Y-Error-Tracking of Robot
        figure5 = plt.figure(figsize=figure_size)
        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.y_error_plot[i])
            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title('Y-Error-Tracking of Robot')
            self._add_title('Y-Error-Tracking of Robot')
            plt.xlabel('time')
            plt.ylabel('y_error (meter)')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            #plt.axis('equal')


        figure6 = plt.figure(figsize=figure_size)
        num_exp_without_ecc = 2
        for i in range(1+num_exp_without_ecc,self.num_exp):
            traj_x, traj_y = zip(*self.interval_control_ecc[i])
            traj_x = self.moving_average(traj_x)
            traj_y = self.moving_average(traj_y)

            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                     linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            plt.xlabel('time')
            plt.ylabel('interval control')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            #plt.axis('equal')

        figure7 = plt.figure(figsize=figure_size)
        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.reward_each_time_interval[i])
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)

            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                     linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title('Reward per episode of Robot')
            self._add_title('Reward per episode of Robot')
            plt.xlabel('time')
            plt.ylabel('Reward per episode')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            #plt.axis('equal')

        figure8 = plt.figure(figsize=figure_size)
        for i in range(1,self.num_exp):
            robot_x_coords, robot_y_coords = zip(*self.pos_error_mag_per_exp[i])

            # Apply moving average filter to smooth the robot path
            robot_x_coords = self.moving_average(robot_x_coords)
            robot_y_coords = self.moving_average(robot_y_coords)
            #plt.plot(robot_x_coords, robot_y_coords, label='Raw Robot Path', linestyle=':', alpha=0.7)
            plt.plot(robot_x_coords, robot_y_coords, label=f'TQC_{self.num_interval_path_planning[i]}',
                      linestyle= linestyle[i-1], color=self.color[i], marker=marker[i-1], markersize = marker_size)
            #plt.title('Error magnitude of Robot')
            plt.xlabel('time')
            plt.ylabel('Error Magnitude')
            #plt.legend()
            self._define_legend_of_plot()
            plt.grid(True)
            #plt.axis('equal')
            self._add_title('Error magnitude of Robot')

        

        plt.show()


    def moving_average(self,data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def _define_legend_of_plot(self):
        plt.legend(
            loc='upper center', # Position at the top center
            bbox_to_anchor=(0.5, 1.15), # Move it above the axes but below the title
            ncol = 7,  # Spread legend items in one horizontal line
            fontsize='small',
            frameon=True,
            fancybox=True,
            shadow=True
        )

    def _add_title(self,title):
        plt.title('', fontsize='small',loc='center')
        #plt.tight_layout()