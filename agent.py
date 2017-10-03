""" Describes the agent that acts in the environment """
class Agent():

    def __init__(self):
        pass

    """ Returns the action to be executed in state """
    def act(self, state):
        pass

    """ Persists a sample to the memory """
    def observe(self, sample):
        pass

    """ Performs replay of memories to improve agent """
    def replay(self):
        pass