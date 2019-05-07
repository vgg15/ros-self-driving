import numpy as np
from collections import deque
import random
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import load_model

class RLQTable:
    def __init__(self, num_states, num_actions, assisted_decay = 0, eps = 0.2):
        self.lambd = 0.95
        self.c = 3
        self.eps = eps
        self.eps_decay = 0.999
        self.lr = 0.8
        self.a = 0
        self.s = 0
        self.assisted_decay = assisted_decay

        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.irand  = 0
        self.iguid  = 0
        self.ilearn = 0
        self.steps  = 0.0

    def Act(self, s, assisted_function=None):
        # get new action
        self.c *= self.assisted_decay

        if self.c > 1:
            self.a = assisted_function(s)
            self.iguid +=1
        elif self.c > 1 or np.random.random() < self.eps:
            self.a = np.random.randint(0, self.num_actions)
            self.irand +=1
        else:
            self.a = np.argmax(self.q_table[s, :])
            self.ilearn +=1

        self.eps *= self.eps_decay
        self.s = s
        self.steps += 1
        return self.a

    def Learn(self, new_s, r):
        self.q_table[self.s, self.a] += r + self.lr * (self.lambd * np.max(self.q_table[new_s, :]) - self.q_table[self.s, self.a])

    def GetTable(self):
        print("Number of steps: ")
        print("Assisted {:.1f}%, Random {:.1f}%, Learned {:.1f}%".format(self.iguid/self.steps*100, self.irand/self.steps*100, self.ilearn/self.steps*100))
        return self.q_table

class RLNN:
    def __init__(self, num_states, num_actions, modelname, assisted_decay = 0, eps = 1):
        self.lambd = 0.99
        self.c = 3
        self.initial_eps = eps
        self.eps = self.initial_eps
        self.eps_decay = 0.9
        self.lr = 0.8
        self.a = 0
        self.s = 0
        self.sample_batch_size = 32

        self.assisted_decay = assisted_decay
        self.q_table = np.zeros((num_states, num_actions))
        self.num_states = num_states
        self.num_actions = num_actions
        self.irand  = 0
        self.iguid  = 0
        self.ilearn = 0
        self.steps  = 0.0
        self.memory = deque(maxlen=1000)
        
        if self.assisted_decay == 0:
            self.epochs = 1
        else:
            self.epochs = 50

        try:
            self.trainModel = self.LoadModel(modelname)
        except:
            self.trainModel = self.createModel()

        self.targetModel = self.createModel()
        self.targetModel.set_weights(self.trainModel.get_weights())
        
    def createModel(self):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_dim=self.num_states))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mae'])

        return model

    def LoadModel(self, modelname):
        model = load_model(modelname)
        print("Loading existing model {}".format(modelname))
        return model

    def SaveModel(self, name):
        self.trainModel.save(name)

    def Sync(self):
        self.targetModel.set_weights(self.trainModel.get_weights())
        #self.eps *= self.eps_decay
        self.eps -= 0.05

    def Reset(self):
        self.eps = self.initial_eps
 
    def Act(self, s, assisted_function = None):
        self.eps = max(0.01, self.eps)

        if np.random.rand(1) < self.eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action=np.argmax(self.trainModel.predict(s)[0])

        return action

    def Learn(self):
        if len(self.memory) < self.sample_batch_size:
            return
        
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        states, actions, rewards, new_states, dones = zip(*sample_batch)
        
        states      = np.array(states).reshape(self.sample_batch_size, self.num_states)
        new_states  = np.array(new_states).reshape(self.sample_batch_size, self.num_states)
        actions     = np.array(actions) #.reshape(self.sample_batch_size,)
        rewards     = np.array(rewards) #.reshape(self.sample_batch_size,)
        dones       = np.array(dones) #.reshape(self.sample_batch_size,)

        targets = self.trainModel.predict(states)
        new_state_targets = self.targetModel.predict(new_states)
        
        Q_futures = new_state_targets.max(axis = 1)

        targets[(np.arange(self.sample_batch_size), actions.astype(int))] = rewards * dones + (rewards + Q_futures * self.lambd)*(~dones)

        self.trainModel.fit(states, targets, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


