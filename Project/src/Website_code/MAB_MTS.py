import numpy as np
import matplotlib.pyplot as plt



T = 500 #number of rounds
K = 9 #number of arms
M = 10 #support size of the distributions

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

 alpha = np.ones((K, M))


 for t in range(T):

  L = np.zeros((K,M))

  for k in range(K):

   L[k, :] = np.random.dirichlet(alpha[k, :])

  I = L.dot(weights.T)
  It = np.argmax(I)

  rt = np.argmax(np.random.multinomial(1, p[It, :], size=1))

  alpha[It, rt] += 1


  regret_total[t] -= expectation[It]

print("I am happy")

regret_total = np.cumsum(regret_total / 100)

t = np.arange(1, T+1)
plt.plot(t, regret_total)
plt.ylabel('Cumulative Regret')
plt.xlabel('Time')
plt.title('The Cumulative Regret of multinomialTS over 100 trials with 9 arms and 500 rounds')
plt.show()
