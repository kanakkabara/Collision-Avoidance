#!/usr/bin/env python
import numpy as np
import random
from collections import deque

class ExperienceReplayBuffer():
    # Experience replay buffer, returns batch_size episodes, each of trace_length
    def __init__(self, batch_size, trace_length, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))   # Actual buffer
        self.episode_buffer = []                    # Episodic buffer
        self.batch_size = batch_size
        self.trace_length = trace_length

    def add_step(self, step):
        # The results of each step are first added to the episodic buffer
        self.episode_buffer.append(np.reshape(np.array(step), [1,5]))

    def flush(self):
        # When gym env returns done or we cross the max number of steps, the episodic buffer is flushed into the actual replay buffer
        self.buffer.append(list(zip(np.array(self.episode_buffer))))
        self.episode_buffer = []
            
    def sample(self):
        traces = []
        for episode in random.sample(self.buffer, self.batch_size):
            # Pick a random point to start the trace
            point = np.random.randint(0, len(episode) + 1 - self.trace_length)
            # Add a trace of length trace_length to be returned
            traces.append(episode[point: point + self.trace_length])
        
        # Extract the states, actions, rewards, next states and dones from the traces
        batch = np.reshape(np.array(traces), [self.batch_size * self.trace_length, 5])
        states = np.vstack(batch[:, 0]/255.0)
        actions = batch[:, 1]
        rewards = batch[:, 2]
        next_states = np.vstack(batch[:, 3]/255.0)
        dones = batch[:, 4]

        return states, actions, rewards, next_states, dones