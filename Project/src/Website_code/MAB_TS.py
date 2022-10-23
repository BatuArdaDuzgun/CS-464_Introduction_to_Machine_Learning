import matplotlib.pyplot as plt
import numpy as np


number_of_clusters = 9

#generating random arms
p = np.zeros(number_of_clusters)
for k in range(number_of_clusters):
    p[k] = np.random.uniform(0, 1)

bestarm = max(p)

T = 500

regret_total = [bestarm * 100] * T
regret_total = np.array(regret_total)

for trial in range(100):

    x = np.zeros(T)

    theta = np.zeros(number_of_clusters)
    alpha = np.ones(number_of_clusters)
    beta = np.ones(number_of_clusters)

    for t in range(T):

        for k in range(number_of_clusters):

            theta[k] = np.random.beta(alpha[k], beta[k])

        x[t] = theta.argmax()
        index = int(x[t])
        rt = np.random.binomial(1, p[index])
        alpha[index] += rt
        beta[index] += (1 - rt)
        regret_total[t] -= p[index]

regret_total = np.cumsum(regret_total / 100)

t = np.arange(1, T + 1)
plt.plot(t, regret_total)
plt.ylabel('Cumulative Regret')
plt.xlabel('Time')
plt.title('The Cumulative Regret of TS over 100 trials with 9 arms and 500 rounds')
plt.show()


