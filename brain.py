""" Contains the agent's logic """
import keras

class Brain():

    def __init__(self):
        model = Sequential()
        
        model.add(Dense(output_dim=64, activation='relu', input_dim=stateCount))
        model.add(Dense(output_dim=actionCount, activation='linear'))
 
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        
        self.model = model

    """ Predicts the Q function values for the given state """
    def predict(self, state):
        self.model.predict(state)

    """ Performs a training iteration with the given batch """
    def train(self, x, y):
        self.model.fit(x, y, batch_size=64)