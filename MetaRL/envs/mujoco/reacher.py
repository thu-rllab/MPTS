import torch
import numpy as np
from gym.envs.mujoco import ReacherEnv as ReacherEnv_


class ReacherEnv(ReacherEnv_):
    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ]).ravel().astype(np.float32).flatten()


class ReacherPosEnv(ReacherEnv):
    def __init__(self, task={}):
        self._task = task
        self.goal = None if len(task.keys()) == 0 else task['target']
        # self.low = -1.3
        # self.high = 1.3
        self.low = -0.2
        self.high = 0.2
        super(ReacherPosEnv, self).__init__()
        a=1

    def step(self, action):
        target_pos = self.get_body_com("target")
        target_pos[:2] = self.goal
        vec = self.get_body_com("fingertip")-target_pos
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = True if abs(ob[4]-target_pos[0])>0.0001 else False
        return ob, reward, done, dict(reward_goal=reward_dist, reward_ctrl=reward_ctrl, task=self.goal)

    def sample_tasks(self, num_tasks, init_dist='Uniform'):
        tasks = []
        for i in range(num_tasks):
            while True:
                target = self.np_random.uniform(low=self.low, high=self.high, size=2)
                if np.linalg.norm(target) < 2:
                    break
            tasks.append({'target': target})
        return tasks
    
    def grid_sample_tasks(self, num_tasks, init_dist='Uniform'):
        goal_1 = np.linspace(self.low, self.high, int(num_tasks**0.5+1))
        goal_2 = np.linspace(self.low, self.high, int(num_tasks**0.5+1))
        tasks = []
        for g1 in goal_1:
            for g2 in goal_2:
                tasks.append({'target': np.array([g1, g2])})
        return tasks
    
    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        if self.goal is None:
            while True:
                self.goal = self.np_random.uniform(self.low, self.high, size=2)
                if np.linalg.norm(self.goal) < 2:
                    break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_task(self, task):
        self._task = task
        self.goal = task['target']
