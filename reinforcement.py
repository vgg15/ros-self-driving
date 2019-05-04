import numpy as np

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

    def Act(self, s, assisted_function):
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
