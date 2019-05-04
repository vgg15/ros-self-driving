import numpy as np
import random

class RLQTable:
    def __init__(self, num_states, num_actions):
        self.lambd = 0.95
        self.eps = 0.5
        self.decay_factor = 0.999
        self.lr = 0.8
        self.a = 0
        self.s = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))

    def Act(self, s):
        # get new action
        if np.random.random() < self.eps:
            self.a = np.random.randint(0, self.num_actions)
        else:
            self.a = np.argmax(self.q_table[s, :])
        self.eps *= self.decay_factor
        self.s = s
        return self.a

    def Learn(self, new_s, r):
        self.q_table[self.s, self.a] += r + self.lr * (self.lambd * np.max(self.q_table[new_s, :]) - self.q_table[self.s, self.a])

    def GetTable(self):
        return self.q_table
