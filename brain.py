""" Contains the agent's logic """
from keras.layers import Dense, Activation, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop


class Brain():

    def __init__(self, actionCount, stateCount, batch_size):
        self.stateCount = stateCount
        self.actionCount = actionCount
        self.batch_size = batch_size
        model = Sequential()

        model.add(Dense(output_dim=16, activation='relu', input_dim=stateCount))
        model.add(Dense(16, activation='relu'))
        #model.add(Dense(48, activation='relu'))
        #model.add(Dense(48, activation='relu'))
        #model.add(Dense(48, activation='relu'))
        model.add(Dense(output_dim=actionCount, activation='linear'))
        #opt = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        opt = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=opt)
        
        self.model = model

    """ Predicts the Q function values for the given state """
    def predict(self, state):
        return self.model.predict(state)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.stateCount)).flatten()

    """ Performs a training iteration with the given batch """
    def train(self, x, y):
        self.model.fit(x, y, batch_size=self.batch_size, epochs=1, verbose=0)