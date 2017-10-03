""" Describes the agent that acts in the environment """
import random
import numpy as np
from brain import Brain
from memory import Memory

class Agent():

    def __init__(self, epsilon, n_actions, n_states, gamma):
        self.epsilon = epsilon
        self.actionCount = n_actions
        self.stateCount = n_states
        self.brain = Brain()
        self.memory = Memory(100000)
        self.batch_size = 64
        self.gamma = gamma

    """ Returns the action to be executed in state """
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            np.argmax(self.brain.predict(state))

    """ Persists a sample to the memory """
    def observe(self, sample):
        self.memory.add(sample)

    """ Performs replay of memories to improve agent """
    def replay(self):
        x = []
        y = []
        batch = self.memory.sample(self.batch_size)

        no_state = np.zeros(self.stateCount)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
 
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        for i in range(self.batch_size):
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