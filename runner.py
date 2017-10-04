""" """
import gym
import h5py
import numpy as np
from environment import Environment
from agent import Agent
from collections import deque

PROBLEM = 'CartPole-v0'
EPISODES = 10000


envi = Environment(PROBLEM)
score = deque(maxlen=EPISODES)

        
try:
    for epi in range(EPISODES):
        [reward, steps] = envi.run()
        score.append(steps)
        if epi % 100 == 0 and epi > 0:
            mean_score = np.mean(score)
            #print("Reward:",reward)
            print("Episode ",epi, " mean score:", mean_score)

finally:
    envi.agent.brain.model.save("cartpole-basic.h5")