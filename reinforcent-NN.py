import numpy as np
import random
import keras
from keras.layers import *
from keras.models import Sequential

# now execute the q learning

def encodeLabels(labels, num_classes):
    LABELS_MIN = -1
    LABELS_MAX = 1

    y = np.zeros(num_classes)
    idx = np.interp(labels, [LABELS_MIN, LABELS_MAX], [0, num_classes-1])
    y[int(round(idx))] = 1

    return y

def main():
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    lr = 0.8
    num_states = 5
    num_actions = 5
    hidden_units = num_states*2

    target_state = 2
    rewards = [0, 0, 5, 0, 0]
    correction = np.linspace(-1,1,num_actions)
    q_table = np.zeros((num_states, num_actions))

    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, num_states)))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    s = 0
    a = 0

    # ... get new frame #
    #new_s = decodeImg(frame)
    angle = 0.0
    new_angle = 0.0
    i = 0
    done = False
    steps = 0
    while(done==False):
    #for _ in range(2000):
        # get reward
        new_s = np.argmax(encodeLabels(new_angle, 5))

        r = rewards[new_s]
        print("State: {}, angle: {:.2}, correction: {}, new_angle: {:.1}, reward: {}, i: {}".format(new_s, angle, correction[a], new_angle, r, i))

        # apply reinforcement learning
        target = r + y * np.max(model.predict(np.identity(num_states)[new_s:new_s + 1]))
        target_vec = model.predict(np.identity(num_states)[s:s + 1])[0]
        target_vec[a] = target
        model.fit(np.identity(num_states)[s:s + 1], target_vec.reshape(-1, num_actions), epochs=1, verbose=0)
        q_table[s,a] += r

        s = new_s
        # predict new angle
        # angle = model.predict(frame)

        angle = random.uniform(-1,1)
        s = np.argmax(encodeLabels(angle, 5))

        # get new action
        if np.random.random() < eps:
            a = np.random.randint(0, num_actions)
        else:
            a = np.argmax(model.predict(np.identity(num_states)[s:s + 1]))

        # get corrective angle based on the action
        new_angle = angle + correction[a]
        eps *= decay_factor
        if (i == 10):
            done = True
        if (new_s == target_state):
            i += 1
        else:
            i = 0
        steps += 1
        

    print(q_table)
    idxs = np.argmax(q_table, axis=1)
    print(idxs)
    print(np.sum(abs(idxs-[4,3,2,1, 0])))        
    print("steps {}".format(steps))

if "__main__" == __name__:
    main()