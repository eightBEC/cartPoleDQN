""" Responsible for running episodes in which the agent is acting """
import gym
from agent import Agent
from gym import wrappers


class Environment():

    def __init__(self, PROBLEM):
        env = gym.make(PROBLEM)
        #env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
        self.env = env
        stateCnt  = env.observation_space.shape[0]
        actionCnt = env.action_space.n
        agent = Agent(0.01, 1, actionCnt, stateCnt, 0.99, 0.001)
        self.agent = agent
    
    def run(self):
        state = self.env.reset()
        self.agent.reset_steps()
        R = 0

        while True:
            action = self.agent.act(state)
            new_state, reward, done, _ = self.env.step(action)

            if done:
                new_state = None
            
            self.agent.observe((state, action, reward, new_state, done))
            #self.agent.replay()

            state = new_state
            R += reward

            self.env.render()

            if done: 
                break


        #print("Total reward:", R, "Steps: ", self.agent.steps)
        self.agent.replay()
        return (R,self.agent.steps)
