""" JF """
import random
from collections import deque

""" Contains the agent's memory """
class Memory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = deque(maxlen=self.capacity)

    def add(self, sample):
        self.samples.append(sample)        

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
