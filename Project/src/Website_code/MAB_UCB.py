import numpy as np
import matplotlib.pyplot as plt


T = 500  # number of rounds
K = 9  # number of arms
M = 10  # support size of the distributions

delta = 0.8 # UCB param
UCB = np.zeros(K)

regret_total = np.zeros(T)

weights = np.arange(M + 1) / M
weights = weights[1:]

p = np.zeros((K, M))

for i in range(K):
 vector = np.random.rand(M)
 vector = np.power(vector, 25) # to make the values more extreme
 temp_sum = sum(vector)
 vector = vector / temp_sum
 p[i, :] = vector

expectation = p.dot(weights.T)

bestarm = max(expectation)

regret_total = [bestarm * 100] * T
regret_total = np.array(regret_total)

for trial in range(100):

    arms = np.zeros(K)
    rewards = np.zeros(K)

    for t in range(K): # initial
        It = t
        rt = (np.argmax(np.random.multinomial(1, p[It], size=1)) + 1) / M
        rewards[It] += rt
        arms[It] += 1

        regret_total[t] -= expectation[It]


    for t in range(K, T):

        for k in range(K):

            UCB[k] = rewards[k] / arms[k] + np.sqrt(2 * np.log(1 / delta) / arms[k])

        It = np.argmax(UCB)
        rt = (np.argmax(np.random.multinomial(1, p[It], size=1)) + 1) / M
        arms[It] += 1
        rewards[It] += rt

        regret_total[t] -= expectation[It]


regret_total = np.cumsum(regret_total / 100)

t = np.arange(1, T + 1)
plt.plot(t, regret_total)
plt.ylabel('Cumulative Regret')
plt.xlabel('Time')
plt.title('The Cumulative Regret of UCB over 100 trials with 9 arms and 500 rounds')
plt.show()


print("I am happy")
