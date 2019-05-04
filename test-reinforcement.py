import numpy as np
import random
import reinforcement as rl

def encodeLabels(labels, num_classes):
    LABELS_MIN = -1
    LABELS_MAX = 1

    y = np.zeros(num_classes)
    idx = np.interp(labels, [LABELS_MIN, LABELS_MAX], [0, num_classes-1])
    y[int(round(idx))] = 1

    return y

def af(s):
    return -s+4

def main():
    num_states = 8
    num_actions = 5
    correction = [-2, -1, 0, 1, 2]#np.linspace(-1,1,num_actions)

    rewards = [0, 2, 5, 2, 0] # TODO: Generalize this
    steps = 0
    num_steps = 5
    for j in range(num_steps):
        new_s = 0
        i = 0
        s = 0
        a = 0
        target_state = 2
        done = False
        #agent = rl.RLQTable(num_states,num_actions, 0.9, eps = 0.2)
        agent = rl.RLNN(num_states,num_actions, 0.92, eps = 0.2)
        print("Run {}/{}".format(j,num_steps))

        while(done==False):
            # get reward
            #new_s = new_angle

            #r = rewards[new_s]

            # apply reinforcement learning
            #q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            #q_table[s, a] += r
            #s = new_s
            # predict new angle
            # angle = model.predict(frame)
            #r = rewards[new_s]
            r = (new_s == target_state)*1
            agent.Learn(new_s, r)
            #print("State: {}, angle: {:.2}, correction: {}, new_angle: {:.1}, reward: {}, i: {}".format(new_s, angle, correction[a], new_angle, r, i))
            
            #angle = random.uniform(-1,1)
            #s = np.argmax(encodeLabels(angle, 5))
            s = random.randint(0,4)
            a = agent.Act(s, af)

            # get corrective angle based on the action
            new_s = s + correction[a]
            if (i == 15):
                done = True
            if (new_s >= target_state-1 and new_s <= target_state+1):
                i += 1
            else:
                i = 0
            steps += 1
            #print("new_s {}, i {} ".format(new_s, i))

    q_table = agent.GetTable()
    print(q_table)
    idxs = np.argmax(q_table, axis=1)
    print(idxs)
    #print(np.sum(abs(idxs-[4,3,2,1, 0])))
    print("Average steps {}".format(steps/num_steps))

if "__main__" == __name__:
    main()