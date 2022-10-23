import numpy as np
import matplotlib.pyplot as plt

T = 500  # number of rounds
K = 9  # number of arms
M = 10  # support size of the distributions

regret_total = np.zeros(T)

weights = np.arange(M + 1) / M
weights = weights[1:]

p = np.zeros((K, M))

for i in range(K):
    vector = np.random.rand(M)
    vector = np.power(vector, 25)  # to make the values more extreme
    temp_sum = sum(vector)
    vector = vector / temp_sum
    p[i, :] = vector

expectation = p.dot(weights.T)

bestarm = max(expectation)

regret_total = [bestarm * 100] * T
regret_total_MTS = np.array(regret_total)
regret_total_UCB = np.array(regret_total)

for trial in range(100):

    alpha = np.ones((K, M))

    for t in range(T):

        L = np.zeros((K, M))

        for k in range(K):
            L[k, :] = np.random.dirichlet(alpha[k, :])

        I = L.dot(weights.T)
        It = np.argmax(I)

        rt = np.argmax(np.random.multinomial(1, p[It, :], size=1))

        alpha[It, rt] += 1

        regret_total_MTS[t] -= expectation[It]

print("I am happy")

regret_total_MTS = np.cumsum(regret_total_MTS / 100)

delta = 0.8  # UCB param
UCB = np.zeros(K)

for trial in range(100):

    arms = np.zeros(K)
    rewards = np.zeros(K)

    for t in range(K):  # initial
        It = t
        rt = (np.argmax(np.random.multinomial(1, p[It], size=1)) + 1) / M
        rewards[It] += rt
        arms[It] += 1

        regret_total_UCB[t] -= expectation[It]

    for t in range(K, T):

        for k in range(K):
            UCB[k] = rewards[k] / arms[k] + np.sqrt(2 * np.log(1 / delta) / arms[k])

        It = np.argmax(UCB)
        rt = (np.argmax(np.random.multinomial(1, p[It], size=1)) + 1) / M
        arms[It] += 1
        rewards[It] += rt

        regret_total_UCB[t] -= expectation[It]

regret_total_UCB = np.cumsum(regret_total_UCB / 100)

t = np.arange(1, T + 1)
plt.plot(t, regret_total_MTS)
plt.plot(t, regret_total_UCB)
plt.ylabel('Cumulative Regret')
plt.xlabel('Time')
plt.legend(["multinomialTS", "UCB"])
plt.title('The Cumulative Regret of UCB and multinomialTS over 100 trials with ' + str(K) + ' arms and ' + str(T) + ' rounds')
plt.show()

print("I am happy")
