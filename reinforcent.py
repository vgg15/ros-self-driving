import numpy as np
import random
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

    rewards = [0, 0, 1, 0, 0]
    correction = np.linspace(-1,1,num_actions)
    q_table = np.zeros((num_states, num_actions))

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
    #for _ in range(20000):
        # get reward
        new_s = np.argmax(encodeLabels(new_angle, 5))

        r = rewards[new_s]
        print("State: {}, angle: {:.2}, correction: {}, new_angle: {:.1}, reward: {}, i: {}".format(new_s, angle, correction[a], new_angle, r, i))

        # apply reinforcement learning
        q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
        #q_table[s, a] += r 
        s = new_s
        # predict new angle
        # angle = model.predict(frame)

        angle = random.uniform(-1,1)
        s = np.argmax(encodeLabels(angle, 5))

        # get new action
        if np.random.random() < eps or np.sum(q_table[s,:]) == 0:
            a = np.random.randint(0, num_actions)
        else:
            a = np.argmax(q_table[s, :])

        # get corrective angle based on the action
        new_angle = angle + correction[a]
        eps *= decay_factor
        if (i == 10):
            done = True
        if (r == 1):
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