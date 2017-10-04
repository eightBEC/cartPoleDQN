""" Describes the agent that acts in the environment """
import random
import numpy as np
from brain import Brain
from memory import Memory
from math import exp

class Agent():

    def __init__(self, min_epsilon, max_epsilon, n_actions, n_states, gamma, lmbda):
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon
        self.actionCount = n_actions
        self.stateCount = n_states
        self.batch_size = 64
        self.brain = Brain(n_actions, n_states, self.batch_size)
        self.memory = Memory(100000)
        self.steps = 0
        self.gamma = gamma
        self.lmbda = lmbda

    """ Returns the action to be executed in state """
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(state))

    """ Persists a sample to the memory """
    def observe(self, sample):
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * exp(-self.lmbda * self.steps)

    def reset_steps(self):
        self.steps = 0

    """ Performs replay of memories to improve agent """
    
    def replay(self):
        
        batch = self.memory.sample(self.batch_size)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCount)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
 
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = np.zeros((self.batch_size, self.stateCount))
        y = np.zeros((self.batch_size, self.actionCount))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
 
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(p_[i])
    
            x[i] = s
            y[i] = t
        
        self.brain.train(x,y)
