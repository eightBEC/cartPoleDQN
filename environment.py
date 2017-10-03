""" Responsible for running episodes in which the agent is acting """
import gym
from agent import Agent

class Environment():

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        PROBLEM = 'CartPole-v0'
        env = Environment(PROBLEM)
 
        stateCnt  = env.env.observation_space.shape[0]
        actionCnt = env.env.action_space.n
 
        agent = Agent(0.01, actionCnt, stateCnt, 0.99)
 
        while True:
            env.run(agent)
    
    def run(self):
        state = self.env.reset()
        while True:
            action = self.agent.act(state)
            new_state, reward, done, info = self.env.step(action)

            if done:
                new_state = None
            
            self.agent.observe((state, action, reward, new_state))
            self.agent.replay()

            state = new_state

            if done: 
                break