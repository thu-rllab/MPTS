import torch
import numpy as np
from gym.envs.mujoco import Walker2dEnv as Walker2dEnv_


class Walker2dEnv(Walker2dEnv_):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), self.get_body_com("torso").flat]).ravel().astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 2800, 1500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            data = np.rot90(np.rot90(data))
            data = np.fliplr(data)

            from PIL import Image, ImageFont, ImageDraw
            data = Image.fromarray(data)
            draw = ImageDraw.Draw(data)
            font = ImageFont.truetype("/System/Library/fonts/SFNSText.ttf", 50)
            # draw.text((x, y),"Sample Text",(r,g,b))

            y_offset = 300
            draw.text((1200, 0+y_offset), "Number of updates: {}".format(self.num_updates), (255, 255, 255), font=font)

            # add task-relevant text
            if self.task_type == 'goal':
                if self.task == -1:
                    draw.text((700, 100+y_offset), "Task: Walk backwards", (255, 255, 255), font=font)
                elif self.task == +1:
                    draw.text((700, 100+y_offset), "Task: Walk forwards", (255, 255, 255), font=font)
                # prediction
                go_left_prob = self.direction_pred
                draw.text((1500, 100+y_offset), "Predictions from context parameters:", (255, 255, 255), font=font)
                if self.task == -1:
                    draw.text((1600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (0, 255, 0), font=font)
                    draw.text((1600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (255, 0, 0), font=font)
                else:
                    draw.text((1600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (255, 0, 0), font=font)
                    draw.text((1600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (0, 255, 0), font=font)

            elif self.task_type == 'vel':
                draw.text((700, 100+y_offset), "Task: Walk at velocity {}".format(self.task), (255, 255, 255), font=font)
                draw.text((700, 170+y_offset), "Current velocity:      {}".format(np.round(float(self.forward_vel), 2)), (255, 255, 255), font=font)

            # add reward text
            draw.text((700, 170+y_offset), "Return (total):   {}".format(np.round(float(self.collected_return), 2)), (255, 255, 255), font=font)
            draw.text((700, 240+y_offset), "Return (forward): {}".format(np.round(float(self.forward_return), 2)), (255, 255, 255), font=font)

            data = np.array(data)

            return data
        elif mode == 'human':
            self._get_viewer().render()

class Walker2dVelEnv(Walker2dEnv):
    def __init__(self, task={}):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        super(Walker2dVelEnv, self).__init__()

    def step(self, action):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        forward_vel = ((posafter - posbefore) / self.dt)
        self.forward_vel = forward_vel
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost + alive_bonus
        done = False #not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, init_dist='Uniform'):
        velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks
    
    def grid_sample_tasks(self, num_tasks, init_dist='Uniform'):
        vel = np.linspace(0.0, 2.0, num_tasks)
        tasks = [{'velocity': velocity} for velocity in vel]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

class Walker2dMassVelEnv(Walker2dEnv):

    def __init__(self, task={}):
        self._task = task
        self.mass_scale_range = [0.75, 1.25]
        self.vel_range = [0.0, 2.0]
        self._goal_vel = task.get('velocity', 0.0)
        super(Walker2dMassVelEnv, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)

    def step(self, action):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        forward_vel = ((posafter - posbefore) / self.dt)
        self.forward_vel = forward_vel
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost + alive_bonus
        done = False #not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
    

    
    def sample_tasks(self, num_tasks, init_dist):
        '''
        Sample envs from the original task dist.
        num_tasks: scalar, batch size of tasks
        init_dist: dist, base dist of the tasks
        '''        
        # types of initial goal distributions: uniform/Gaussian
        if init_dist == 'Uniform':
            mass_scales = self.np_random.uniform(0.75, 1.25, size=(num_tasks,))
            velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        elif init_dist == 'Normal':
            mass_scales = np.random.normal(1.0, 0.08, size=(num_tasks,))
            velocities = np.random.normal(1.0, 0.33, size=(num_tasks,))
            
            # clip the sampled hyper variable into a defined scope
            mass_scales = np.clip(mass_scales, 0.75, 1.25)
            velocities = np.clip(velocities, 0.0, 2.0)
            
        # produce the list of the tuple (mass_scale, velocity)
        mass_vel_tuples = [(mass_scales[i], velocities[i]) for i in range(num_tasks)]
        tasks = [{'m_scale': mass_vel[0],'velocity': mass_vel[1]} for mass_vel in mass_vel_tuples]
        
        return tasks        
    
    def grid_sample_tasks(self, num_tasks, init_dist='Uniform'):
        mass_s = np.linspace(0.75, 1.25, int(num_tasks**0.5+1))
        vel = np.linspace(0.0, 2.0, int(num_tasks**0.5+1))
        tasks = [{'m_scale': mass, 'velocity': velocity} for mass in mass_s for velocity in vel]
        return tasks
    
    def sample_init_param(self, num_tasks, init_dist):
        # sample batch of hyper-params of tasks for the initialization of tasks
        if init_dist == 'Uniform':
            mass_scales = np.random.uniform(self.mass_scale_range[0], self.mass_scale_range[1], num_tasks)
            velocities = np.random.uniform(self.vel_range[0], self.vel_range[1], num_tasks) 
        elif init_dist == 'Normal':
            mass_scales = np.random.normal(1.0, 0.08, num_tasks)
            velocities = np.random.normal(1.0, 0.335, num_tasks)
            mass_scales = np.clip(mass_scales, self.mass_scale_range[0], self.mass_scale_range[1])
            velocities = np.clip(velocities, self.vel_range[0], self.vel_range[1])        
        
        init_param_tensor = torch.zeros(num_tasks, 2)
            
        for i in range(num_tasks):
            init_param_tensor[i, 0] = mass_scales[i]
            init_param_tensor[i, 1] = velocities[i]
    
        return init_param_tensor    
    
    def generate_tasks(self, trasformed_param):
        '''
        Generate envs from the generated hyper variables
        trasformed_param: numpy array, generated from distribution adversary
        '''
        
        tasks = [{'m_scale': mass_vel[0],'velocity': mass_vel[1]} for mass_vel in trasformed_param]
        return tasks 
    
    def reset_task(self, task):
        self._task = task
        
        mass = np.copy(self.original_mass)
        mass *= task['m_scale']
        self.model.body_mass[:] = mass
        self._goal_vel = task['velocity']
        
    def get_sim_params(self):
        # feedback the env hyper variables from the task
        print ('the hyper variable values are tuple of mass and goal velocity')
        
        return (self.model.body_mass[:], self._goal_vel)
