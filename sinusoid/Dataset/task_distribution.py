import torch
import numpy as np
import pdb

#generating sinusoidal data
class SineTask():
    def __init__(self,amp,phase,min_x,max_x):
        self.phase=phase
        self.max_x=max_x
        self.min_x=min_x
        self.amp=amp

    def sample_data(self,size=1):
        x=np.random.uniform(self.max_x,self.min_x,size)
        y=self.true_sine(x)
        x=torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y=torch.tensor(y, dtype=torch.float).unsqueeze(1)
        return x,y

    def true_sine(self,x):
        y=self.amp*np.sin(self.phase+x)
        return y


class SineDistribution():
    def __init__(self, min_amp, max_amp, min_phase, max_phase, min_x, max_x):
        self.min_amp = min_amp
        self.max_phase = max_phase
        self.min_phase = min_phase
        self.max_amp = max_amp
        self.min_x = min_x
        self.max_x = max_x

    def sample_task(self):
        amp = np.random.uniform(self.min_amp, self.max_amp)
        phase = np.random.uniform(self.min_phase, self.max_phase)
        return SineTask(amp, phase, self.min_x, self.max_x)

    def active_sample_task(self, amp, phase):
        return SineTask(amp, phase, self.min_x, self.max_x)

    def task_descriptor_candidate(self, num_candidate):
        # grid search get 100 candidate tasks.
        amp_list_candidate = np.random.uniform(self.min_amp, self.max_amp, size=[num_candidate, 1])
        phase_list_candidate = np.random.uniform(self.min_phase, self.max_phase, size=[num_candidate, 1])
        amp_list_candidate = np.float32(amp_list_candidate)
        phase_list_candidate = np.float32(phase_list_candidate)
        return amp_list_candidate, phase_list_candidate



